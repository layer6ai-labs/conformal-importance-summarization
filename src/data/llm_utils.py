import dspy
import time

from typing import List, Type, cast

api_key_openai = ""
lm_gpt4omini = dspy.LM('openai/gpt-4o-mini', api_key=api_key_openai, max_tokens=10000)

api_key_google = ""
lm_gemini2flashlite = dspy.LM("gemini/gemini-2.0-flash-lite", api_key=api_key_google, max_tokens=10000)
lm_gemini25flash = dspy.LM("gemini/gemini-2.5-flash-preview-04-17", api_key=api_key_google, max_tokens=10000)

lm_qwen38B = dspy.LM(f"ollama_chat/Qwen3:8B", api_base="http://localhost:11434", api_key="", max_tokens=10000)

lm_llama38B = dspy.LM(f"ollama_chat/Llama3:8B", api_base="http://localhost:11434", api_key="", max_tokens=10000)


class SignatureField:
    def __init__(self, name: str, field_type: type, description: str):
        self.name = name
        self.field_type = field_type
        self.description = description


def create_signature_type(
        signature: str,
        instructions: str,
        inputs: List[SignatureField],
        outputs: List[SignatureField]
) -> Type[dspy.Signature]:
    """
    :param signature: Name of the signature.
    :param instructions: Instructions for the signature.
    :param inputs: Dictionary of input fields (field -> description).
    :param outputs: Dictionary of output fields (field -> description).
    """
    input_keys = [field.name for field in inputs]
    if len(input_keys) != len(set(input_keys)):
        raise ValueError("Duplicate input field names found")

    output_keys = [field.name for field in outputs]
    if len(output_keys) != len(set(output_keys)):
        raise ValueError("Duplicate output field names found")

    duplicate_keys = set(input_keys).intersection(output_keys)
    if duplicate_keys:
        raise ValueError(f"Duplicate keys found in inputs and outputs: {duplicate_keys}")

    annotations = {}
    fields = {'__doc__': instructions}

    for item in inputs:
        fields[item.name] = dspy.InputField(description=item.description)
        annotations[item.name] = item.field_type

    for item in outputs:
        fields[item.name] = dspy.OutputField(description=item.description)
        annotations[item.name] = item.field_type

    output = type(signature, (dspy.Signature,), {"__annotations__": annotations, **fields})
    return cast(Type[dspy.Signature], output)


def prepare_label_IO(claims: list[str], summary: str) -> tuple[dict[str, str], list[str]]:
    inputDict = {f"claim_{i + 1}": claim for i, claim in enumerate(claims)}
    inputDict["summary"] = summary
    outputKeys = [f"label_{i + 1}" for i in range(len(claims))]
    return (inputDict, outputKeys)


def create_label_fields(input_dict: dict) -> tuple[list[SignatureField], list[SignatureField]]:
    fields_in = []
    fields_out = []

    for key in input_dict:
        # Define custom descriptions based on the key, if desired.
        if key.startswith("claim_"):
            description = f"{key}."
            field_type = str

            output_name = key.replace("claim", "label")
            fields_out.append(SignatureField(name=output_name, field_type=int, description=f"Label for {key}"))
        elif key == "summary":
            description = "Summary text."
            field_type = str
        else:
            description = f"Field {key}"
            field_type = str  # default type
        fields_in.append(SignatureField(name=key, field_type=field_type, description=description))
    return (fields_in, fields_out)


def createLabelSignature(inputDict):
    # Dynamically create input fields: one for each claim.
    input_fields, output_fields = create_label_fields(inputDict)

    # Use your create_signature_type function to dynamically create the signature class.
    DynamicSignature: Type[dspy.Signature] = create_signature_type(
        signature="EvaluateGroundTruthDynamic",
        instructions=("""Evaluate whether each input claims is included in the summary text. 
        The output labels, corresponding to each input claims, should be either 0 or 1, indicating whether the corresponding claim, 
        or the information it carries, is indeed included in the actual summary.
        For example, if claim_1's information is contained in the summary, then label_1 should be 1; 
        if information carried in claim_3 cannot be found in the summary text, then label_3 should be 0.
        """),
        inputs=input_fields,
        outputs=output_fields
    )
    return DynamicSignature


def extractLabels(claims, summary):
    inputDict, outputKeys = prepare_label_IO(claims, summary)
    gtSignature = createLabelSignature(inputDict)
    evalGT = dspy.Predict(gtSignature)
    try:
        out = evalGT(**inputDict)
        return [out[key] for key in outputKeys]
    except Exception:
        print("For summary '{}...', inference error occurred".format(summary[:10]))
        return None


def addLabel(df, LM, col='input_sentences_labels'):
    dspy.configure(lm=LM)
    df[col] = df.progress_apply(lambda row: extractLabels(row['input_sentences'], row['summary']), axis=1)
    return df


def prepare_importance_IO(claims: list[str], text: str):
    inputDict = {f"claim {i + 1}": claim for i, claim in enumerate(claims)}
    inputDict["text"] = text
    outputKeys = [f"score {i + 1}" for i in range(len(claims))]
    return (inputDict, outputKeys)


def create_importance_fields(input_dict: dict):
    fields_in = []
    fields_out = []
    for key in input_dict:
        # Define custom descriptions based on the key, if desired.
        if key.startswith("claim "):
            description = f"{key}."
            field_type = str

            output_name = key.replace("claim", "score")
            fields_out.append(SignatureField(name=output_name,
                                             field_type=float,
                                             description=f"Importance Score for {key}")
                              )
        elif key == "text":
            description = "Original text."
            field_type = str
        else:
            description = f"Field {key}"
            field_type = str  # default type
        fields_in.append(SignatureField(name=key, field_type=field_type, description=description))
    return fields_in, fields_out


def createImportanceScoreSignature(inputDict):
    # Dynamically create input fields: one for each claim.
    input_fields, output_fields = create_importance_fields(inputDict)

    # Use your create_signature_type function to dynamically create the signature class.
    DynamicSignature: Type[dspy.Signature] = create_signature_type(
        signature="EvaluateImportanceScore",
        instructions=("""Please evaluate the importance of each input claims in the original text, 
        based on how the information carried in the claim is aligned with the overall message. 
        Please provide a importance score for EACH input claim.
        Each output score should be a two decimal float number ranged between 0 and 1, 
        indicating how important the corresponding input claim is in the context of the text document. 
        For example, if claim 1's information is highly aligned with that of the input text, 
        and very likely to be included in the summary, then score 1 should be close to 1, say greater than 0.8; 
        if information carried in claim 3 is trivial or only remotely related to the central message of the text, 
        and is not worthy of inclusion in the summary, then score 3 should be close to 0, say less than 0.2.
        """),
        inputs=input_fields,
        outputs=output_fields
    )
    return DynamicSignature


def extractImportanceScores(claims, text):
    inputDict, outputKeys = prepare_importance_IO(claims, text)
    scoreSignature = createImportanceScoreSignature(inputDict)
    evalScore = dspy.Predict(scoreSignature)
    attempt = 0
    cooldown = 20
    while attempt < 5:
        try:
            out = evalScore(**inputDict)
            return [out[key] for key in outputKeys]
        except Exception as e:
            attempt += 1
            last_exception = e
            if attempt >= 5:
                break
            print("Attempt {}/5".format(attempt))
            time.sleep(cooldown * attempt)
    print("All re-tries failed")
    print("For entry '{}...', inference error occurred: {}".format(text[:20], last_exception))
    return None


def addImportanceScores(df, LM, col='gpt_scores'):
    dspy.configure(lm=LM)
    df[col] = df.progress_apply(lambda row: extractImportanceScores(row['input_sentences'], row['input']), axis=1)
    return df


def prepare_binary_IO(claims: list[str], text: str):
    inputDict = {f"claim {i + 1}": claim for i, claim in enumerate(claims)}
    inputDict["text"] = text
    outputKeys = [f"score {i + 1}" for i in range(len(claims))]
    return (inputDict, outputKeys)


def create_binary_fields(input_dict: dict):
    fields_in = []
    fields_out = []
    for key in input_dict:
        # Define custom descriptions based on the key, if desired.
        if key.startswith("claim "):
            description = f"{key}."
            field_type = str

            output_name = key.replace("claim", "score")
            fields_out.append(SignatureField(name=output_name,
                                             field_type=int,
                                             description=f"Importance Score for {key}")
                              )
        elif key == "text":
            description = "Original text."
            field_type = str
        else:
            description = f"Field {key}"
            field_type = str  # default type
        fields_in.append(SignatureField(name=key, field_type=field_type, description=description))
    return (fields_in, fields_out)


def createBinaryImportanceSignature(inputDict):
    # Dynamically create input fields: one for each claim.
    input_fields, output_fields = create_binary_fields(inputDict)

    # Use your create_signature_type function to dynamically create the signature class.
    DynamicSignature: Type[dspy.Signature] = create_signature_type(
        signature="EvaluateBinaryImportanceScore",
        instructions=("""Evaluate the importance of each input claims in the original text, 
        based on how the information carried in the claim is aligned with the overall message. 
        Please provide a binary importance score for EACH input claim.
        Each output score should be either 0 or 1, 
        indicating whether the corresponding input claim is important enough in the context of the text document to be included in the summary. 
        For example, if claim 1's information is highly aligned with that of the input text, 
        and very likely to be included in the summary, then score 1 should be 1; 
        if information carried in claim 3 is trivial or only remotely related to the central message of the text, 
        and is not worthy of inclusion in the summary, then score 3 should be 0.
        """),
        inputs=input_fields,
        outputs=output_fields
    )
    return DynamicSignature


def extractBinaryScores(claims, text):
    inputDict, outputKeys = prepare_binary_IO(claims, text)
    scoreSignature = createBinaryImportanceSignature(inputDict)
    evalScore = dspy.Predict(scoreSignature)
    attempt = 0
    cooldown = 20
    while attempt < 5:
        try:
            out = evalScore(**inputDict)
            return [out[key] for key in outputKeys]
        except Exception as e:
            attempt += 1
            last_exception = e
            if attempt >= 5:
                break
            print("Attempt {}/5".format(attempt))
            time.sleep(cooldown)
    print("All re-tries failed")
    print("For entry '{}...', inference error occurred: {}".format(text[:20], last_exception))
    return None


def addBinaryScores(df, LM, col='gpt_scores_binary'):
    dspy.configure(lm=LM)
    df[col] = df.progress_apply(lambda row: extractBinaryScores(row['input_sentences'], row['input']), axis=1)
    return df


def addLLMLabelColumn(df, LM=lm_gpt4omini):
    return addLabel(df, LM, col='input_sentences_labels')


def addLLMScoreColumns(df):
    df = addImportanceScores(df, lm_gpt4omini, col='gpt_scores')
    df = addBinaryScores(df, lm_gpt4omini, col='gpt_scores_binary')

    df = addImportanceScores(df, lm_qwen38B, col='llama_scores')
    df = addImportanceScores(df, lm_llama38B, col='qwen3_scores')
    df = addImportanceScores(df, lm_gemini2flashlite, col='gemini2_scores')

    df = addImportanceScores(df, lm_gemini25flash, col='gemini25_scores')
    df = addBinaryScores(df, lm_gemini25flash, col='gemini25_scores_binary')
    return df

