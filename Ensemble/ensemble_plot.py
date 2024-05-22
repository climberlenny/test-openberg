from toolbox.postprocessing import plot_ensemble_mean, plot_contour_ensemble

file = "test_openberg/Ensemble/output/ensemble_W_H_Ca_Co_1000m.nc"

# plot_ensemble_mean(file)

plot_contour_ensemble(file, c=0.2)
