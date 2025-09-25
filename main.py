from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from src.data.loaddata import load_dataset
from src.evals.base import EvalMethodFactory
from src.evals.curve_area import PRAUCEval
from src.scoring.lexrank_method import LexRankScore
from src.scoring.method import ConfidenceMethod

# --- Default configurations ---
ALL_DATASETS = ["ECT", "MTS", "CNNDM", "CSDS", "tldr"]
ALL_METHODS = [
    "gusum", 
    "cosine-centrality", 
    "sentence-centrality", 
    "lexrank", 
    "gpt", 
    "gemini2", # Gemini 2.0 flash
    "gemini25", # Gemini 2.5 flash
    "llama", # Llama 3
    "qwen3"
]
ALL_METRICS = ["sentence_reduction", "precision-recall", "prauc"]

DEFAULT_ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
DEFAULT_BETAS = [0.7, 0.8, 0.9, 0.95]
DEFAULT_RANKS_STR = ["false"]

DEFAULT_CALIBRATION_PLOT_ALPHAS_PARAMS = (0.01, 0.4, 0.01) # min, max, step
DEFAULT_CALIBRATION_PLOT_BETAS = [0.7, 0.8, 0.9, 0.95] # For lines on the plot
DEFAULT_CALIBRATION_PLOT_CAL_SIZES = [0.05, 0.1]
DEFAULT_CALIBRATION_N_SAMPLES = 100

OUTPUT_DIR_DEFAULT = "experiment_results"
RESULTS_CSV_FILENAME = "experiment_summary.csv"
PLOTS_SUBDIR = "plots"

LABEL_TYPE_MAP = {
    "rouge-1": "rouge1_labels",
    "rouge-2": "rouge2_labels",
    "rouge-l": "rougel_labels",
    "cosine-similarity": "input_sentences_labels",
}

# --- Logger setup ---
logger = logging.getLogger(__name__)

def setup_logging(log_level_str: str):
    """Configures basic logging."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info(f"Logging level set to {log_level_str.upper()}")

# --- Argument parsing ---
def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Run conformal LLM experiments.")

    # Experiment selection
    parser.add_argument("--datasets", nargs="+", default=None, help=f"List of datasets to use. Default: all ({', '.join(ALL_DATASETS)})")
    parser.add_argument("--methods", nargs="+", default=None, help=f"List of scoring methods. Default: all ({', '.join(ALL_METHODS)}). Note: 'gpt' can be slow/expensive.")
    parser.add_argument("--metrics", nargs="+", default=None, help=f"List of evaluation metrics. Default: all ({', '.join(ALL_METRICS)})")
    parser.add_argument("--label-type", default="cosine-similarity", help="Label type for datasets with no explicit label. Default: cosine-similarity. Options are \'cosine-similarity\', \'rouge-1\', \'rouge-2\', \'rouge-l\' ")

    # Hyperparameters for point metrics
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS, help=f"List of alpha values. Default: {DEFAULT_ALPHAS}")
    parser.add_argument("--betas", nargs="+", type=float, default=DEFAULT_BETAS, help=f"List of beta values. Default: {DEFAULT_BETAS}")
    parser.add_argument("--ranks", nargs="+", type=str, default=DEFAULT_RANKS_STR, help=f"List of rank options ('true' or 'false'). Default: {DEFAULT_RANKS_STR}")

    # Control flags
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT, help=f"Directory to save results and plots. Default: {OUTPUT_DIR_DEFAULT}")
    parser.add_argument("--force-recompute-scores", action="store_true", help="Force recomputation of scores, ignoring cache.")
    parser.add_argument("--use-cache-dataset", action="store_true", help="Use cached datasets if available (passed to load_dataset as use_cache).")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set logging level.")

    # Calibration plot specific arguments
    parser.add_argument("--run-calibration-plots", action="store_true", help="Generate and save calibration plots.")
    parser.add_argument("--calplot-alpha-min", type=float, default=DEFAULT_CALIBRATION_PLOT_ALPHAS_PARAMS[0], help="Min alpha for calibration plot x-axis.")
    parser.add_argument("--calplot-alpha-max", type=float, default=DEFAULT_CALIBRATION_PLOT_ALPHAS_PARAMS[1], help="Max alpha for calibration plot x-axis.")
    parser.add_argument("--calplot-alpha-step", type=float, default=DEFAULT_CALIBRATION_PLOT_ALPHAS_PARAMS[2], help="Step for alpha range in calibration plot.")
    parser.add_argument("--calplot-betas", nargs="+", type=float, default=DEFAULT_CALIBRATION_PLOT_BETAS, help="Beta values for lines in calibration plots.")
    parser.add_argument("--calplot-cal-sizes", nargs="+", type=float, default=DEFAULT_CALIBRATION_PLOT_CAL_SIZES, help="Calibration sizes for calibration plots.")
    parser.add_argument("--calplot-n-samples", type=int, default=DEFAULT_CALIBRATION_N_SAMPLES, help="Number of samples for calibration plots.")

    return parser

def str_to_bool(value: str) -> bool:
    """Converts string 'true'/'false' to boolean."""
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert '{value}' to boolean. Expected 'true' or 'false'.")

# --- Main experiment logic ---
def run_experiments(args: argparse.Namespace):
    """Runs the main experiment loop based on parsed arguments."""
    overall_start_time = time.time() # Added for overall timing
    
    logger.info("Starting experiment setup...")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured output directory exists: {output_path.resolve()}")
    
    plots_path = output_path / PLOTS_SUBDIR
    if args.run_calibration_plots:
        plots_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured plots subdirectory exists: {plots_path.resolve()}")

    logger.info(f"Results will be saved to: {output_path.resolve()}")
    logger.info(f"Plots will be saved to: {plots_path.resolve()} (if calibration plots are run)")

    # Determine combinations to run
    datasets_to_run = args.datasets if args.datasets else ALL_DATASETS
    methods_to_run = args.methods if args.methods else ALL_METHODS
    metrics_to_run = args.metrics if args.metrics else ALL_METRICS
    ranks_to_run = [str_to_bool(r) for r in args.ranks]

    logger.info(f"Datasets to process: {datasets_to_run}")
    logger.info(f"Methods to evaluate: {methods_to_run}")
    logger.info(f"Metrics to compute: {metrics_to_run}")
    logger.info(f"Rank options: {ranks_to_run}")
    logger.info(f"Alpha values: {args.alphas}")
    logger.info(f"Beta values: {args.betas}")
    if args.force_recompute_scores:
        logger.info("Forcing recomputation of all scores.")
    if args.use_cache_dataset:
        logger.info("Attempting to use cached datasets.")


    all_results_data = []
    logger.info("Experiment setup complete. Starting main loop...")

    for dataset_name in tqdm(datasets_to_run, desc="Datasets"):
        dataset_start_time = time.time()
        logger.info(f"===== Processing dataset: {dataset_name} =====")
        try:
            label_colname = LABEL_TYPE_MAP.get(args.label_type, "input_sentences_labels")
            logger.info(f"Loading calibration data for {dataset_name}...")
            d_cal, d_test = load_dataset(dataset_name, label_col=label_colname)
            logger.info(f"Loaded calibration data for {dataset_name} ({len(d_cal.instances)} instances).")

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}", exc_info=True)
            continue

        for method_name in tqdm(methods_to_run, desc=f"Methods ({dataset_name})", leave=False):
            method_start_time = time.time()
            logger.info(f"--- Using method: {method_name} for dataset: {dataset_name} ---")
            
            scorer_instance: ConfidenceMethod | str
            if method_name == "lexrank":
                try:
                    logger.info(f"Initializing LexRankScore for {dataset_name}...")
                    all_docs_cal = [inst.text for inst in d_cal.instances if inst.text]
                    all_docs_test = [inst.text for inst in d_test.instances if inst.text]
                    all_docs_combined = all_docs_cal + all_docs_test

                    if not all_docs_combined:
                        logger.warning(f"No documents found for LexRank in dataset {dataset_name}. Skipping LexRank for this dataset.")
                        continue
                    
                    logger.debug(f"LexRank: combining {len(all_docs_cal)} cal docs and {len(all_docs_test)} test docs.")
                    scorer_instance = LexRankScore(all_docs_combined)
                    logger.info(f"LexRankScore initialized successfully for {dataset_name} with {len(all_docs_combined)} total documents.")
                except Exception as e:
                    logger.error(f"Failed to initialize LexRankScore for {dataset_name}: {e}", exc_info=True)
                    continue
            else:
                scorer_instance = method_name 
                logger.info(f"Using method name '{method_name}' directly for scoring.")

            for metric_name in tqdm(metrics_to_run, desc=f"Metrics ({dataset_name}/{method_name})", leave=False):
                logger.info(f"- Calculating metric: {metric_name} for {dataset_name}/{method_name} -")

                if metric_name == "prauc":
                    metric_prauc_start_time = time.time()
                    logger.info(f"Processing PRAUC: iterating over {len(ranks_to_run)} rank option(s).")
                    for rank in tqdm(ranks_to_run, desc="Rank (PRAUC)", leave=False):
                        current_params = {
                            "dataset": dataset_name, "method": method_name, "metric_class": metric_name,
                            "alpha": None, "beta": None, "rank": rank
                        }
                        logger.debug(f"PRAUC evaluation with params: rank={rank}")
                        try:
                            logger.info(f"Computing PRAUC for {dataset_name}/{method_name} (rank={rank})...")
                            eval_obj = PRAUCEval(scorer_instance, d_test)
                            eval_results = eval_obj.compute()
                            logger.info(f"PRAUC computation complete for {dataset_name}/{method_name} (rank={rank}). Results: {eval_results}")
                            for res_key, res_val in eval_results.items():
                                all_results_data.append({**current_params, "result_metric_name": res_key, "result_value": res_val})
                        except Exception as e:
                            logger.error(f"Error during PRAUC evaluation for {current_params}: {e}", exc_info=True)
                    metric_prauc_end_time = time.time()
                    logger.info(f"Time taken for PRAUC (method {method_name}, dataset {dataset_name}): {metric_prauc_end_time - metric_prauc_start_time:.2f} seconds.")
            
                elif metric_name in ["sentence_reduction", "precision-recall"]:
                    metric_point_start_time = time.time()
                    logger.info(f"Processing {metric_name}: iterating over {len(args.alphas)} alpha(s), {len(args.betas)} beta(s), {len(ranks_to_run)} rank option(s).")
                    for alpha in tqdm(args.alphas, desc="Alpha", leave=False):
                        for beta in tqdm(args.betas, desc="Beta", leave=False):
                            for rank_val in tqdm(ranks_to_run, desc="Rank", leave=False): 
                                current_params = {
                                    "dataset": dataset_name, "method": method_name, "metric_class": metric_name,
                                    "alpha": alpha, "beta": beta, "rank": rank_val
                                }
                                logger.debug(f"{metric_name} evaluation with params: alpha={alpha}, beta={beta}, rank={rank_val}")
                                try:
                                    logger.info(f"Computing {metric_name} for {dataset_name}/{method_name} (alpha={alpha}, beta={beta}, rank={rank_val})...")
                                    eval_obj = EvalMethodFactory.create(
                                        metric_name, scorer_instance, d_test, d_cal,
                                        alpha=alpha, beta=beta,
                                        rank=rank_val
                                    )
                                    eval_results = eval_obj.compute()
                                    logger.info(f"{metric_name} computation complete for {dataset_name}/{method_name} (params above). Results: {eval_results}")
                                    for res_key, res_val in eval_results.items():
                                        all_results_data.append({**current_params, "result_metric_name": res_key, "result_value": res_val})
                                except Exception as e:
                                    logger.error(f"Error during {metric_name} evaluation for {current_params}: {e}", exc_info=True)
                    metric_point_end_time = time.time()
                    logger.info(f"Time taken for {metric_name} (method {method_name}, dataset {dataset_name}): {metric_point_end_time - metric_point_start_time:.2f} seconds.")
                else:
                    logger.warning(f"Metric '{metric_name}' is not recognized for point evaluation. Skipping.")
            logger.info(f"Finished all metrics for {dataset_name}/{method_name}.")

            if args.run_calibration_plots:
                plot_generation_start_time = time.time()
                logger.info(f"--- Generating calibration plots for {dataset_name}/{method_name} ---")
                calplot_alphas_range = np.arange(args.calplot_alpha_min, args.calplot_alpha_max + args.calplot_alpha_step, args.calplot_alpha_step)
                logger.debug(f"Calibration plot alpha range: min={args.calplot_alpha_min}, max={args.calplot_alpha_max}, step={args.calplot_alpha_step} -> {len(calplot_alphas_range)} points")
                logger.debug(f"Calibration plot beta values for lines: {args.calplot_betas}")
                logger.debug(f"Calibration plot cal_sizes: {args.calplot_cal_sizes}")
                logger.debug(f"Calibration plot n_samples: {args.calplot_n_samples}")
                
                method_to_plot = method_name

                for rank_plot in tqdm(ranks_to_run, desc="Rank (CalPlot)", leave=False):
                    for cal_size_val in tqdm(args.calplot_cal_sizes, desc="CalSize (CalPlot)", leave=False):
                        fig = None 
                        logger.info(f"Generating calibration plot for {dataset_name}/{method_to_plot} (rank={rank_plot}, cal_size={cal_size_val})...")
                        try:
                            logger.debug(f"Calling d_test.plot_calibration with method='{method_to_plot}', rank={rank_plot}, cal_size={cal_size_val}")
                            fig, ax = d_test.plot_calibration(
                                method=method_to_plot, 
                                ranking=rank_plot,
                                alphas=calplot_alphas_range,
                                betas=args.calplot_betas, 
                                cal_size=cal_size_val,
                                n_samples=args.calplot_n_samples
                            )
                            ax.set_title(f"Calibration Plot: {dataset_name}/{method_to_plot}\nRank: {rank_plot}, Cal. Size: {cal_size_val}")
                            plot_filename = f"{dataset_name}_{method_to_plot}_rank{rank_plot}_calsize{cal_size_val}_calibration.png"
                            fig.tight_layout()
                            fig.savefig(plots_path / plot_filename)
                            plt.close(fig)
                            fig = None
                            logger.info(f"Saved calibration plot: {plots_path / plot_filename}")
                        except Exception as e:
                            logger.error(f"Failed to generate calibration plot for {dataset_name}/{method_to_plot} (rank={rank_plot}, cal_size={cal_size_val}): {e}", exc_info=True)
                            if fig is not None:
                                plt.close(fig)
                logger.info(f"Finished calibration plots for {dataset_name}/{method_name}.")
                plot_generation_end_time = time.time()
                logger.info(f"Time taken for calibration plots (method {method_name}, dataset {dataset_name}): {plot_generation_end_time - plot_generation_start_time:.2f} seconds.")
            else:
                logger.debug(f"Skipping calibration plots for {dataset_name}/{method_name} as --run-calibration-plots is not set.")
            
            method_end_time = time.time()
            logger.info(f"Time taken for method {method_name} on dataset {dataset_name}: {method_end_time - method_start_time:.2f} seconds.")
        logger.info(f"===== Finished processing dataset: {dataset_name} =====")
        dataset_end_time = time.time()
        logger.info(f"Time taken for dataset {dataset_name}: {dataset_end_time - dataset_start_time:.2f} seconds.")
    
    logger.info("All datasets and methods processed.")
    if all_results_data:
        logger.info(f"Saving {len(all_results_data)} result entries to CSV...")
        results_df = pd.DataFrame(all_results_data)
        csv_path = output_path / RESULTS_CSV_FILENAME
        results_df.to_csv(csv_path, index=False)
        logger.info(f"All experiment results saved to {csv_path.resolve()}")
    else:
        logger.info("No results were generated to save.")

    overall_end_time = time.time()
    logger.info(f"Total experiment duration: {overall_end_time - overall_start_time:.2f} seconds.")
    logger.info("Experiment run completed successfully.")

if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger.debug("Command-line arguments parsed and logging configured.")
    
    logger.info("Starting experiment run with the following arguments:")
    for arg_name, value in sorted(vars(args).items()):
        logger.info(f"  {arg_name}: {value}")
        
    run_experiments(args)
