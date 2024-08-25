mlr.master.RData contains mlr.master dataframe (the original data source for the analysis).

mlr-master_calculate-synergy.Rmd contains full code used to generate the two csv files.

mlr-master_drug-pairs_subset.csv shows summarized results of drug combinations per unique donor pair.
* block_id is a unique identifier for each donor-pair drug combination across all experiments (derived using a combination of assay number, donor IDs , analytes, and drug combos)
* assay_type lists the experiment type, can be thought of as a cell line
* assay_number lists the experiment ID
* drug1 and drug2 lists assets used in the experiment
* analyte lists cytokine name (filtered to IFNg, IL-2, and TNF-a)
* donor_1_id, donor_2_id are the cell donor IDs used in the experiment (since synergy results can vary donor to donor)
* mean_HSA_synergy lists the mean synergy score for the drug combinations (monotherapy observations are excluded). This data was already calculated by the R package, so I did not separately calculate the median.

mlr-master_synergy-scores_subset.csv
* block_id, assay_type, assay_number, drug1, drug2, analyte, donor_1_id, and donor_2_id are same as above
* conc1 and conc2 are the drug concentrations used for the corresponding drug
* response_readout is the measured cytokine results (in pg/mL), with values corrected for no-treatment controls. This value would probably be used for fold change calculations instead of response_induction.
* response_induction is a percent-scaled value by experiment, equal to response_readout / max(response_readout of corresponding experiment). This value was used for synergy calculations.
* HSA_synergy lists the calculated synergy score for the drug combination at their respective concentrations

fold-change calculation thoughts, using mlr-master_synergy-scores:
for each block_id,
	- numerator: determine mean/median response_readout for the drug combo rows (conc1 != 0 or conc2 != 0)
	- denominators: determine mean/median response_readout for each drug as a monotherapy (conc1 == 0) (conc2 == 0)
	- calculate fold-change for each drug in the combination (n = 2)

The full results of the synergy analysis with all the different synergy models and cytokines are contained within mlr-master_drug-pairs.csv and mlr-master_synergy-scores.csv

resource on synergyfinder package: https://www.bioconductor.org/packages/release/bioc/vignettes/synergyfinder/inst/doc/User_tutorual_of_the_SynergyFinder_plus.html#data-with-replicates