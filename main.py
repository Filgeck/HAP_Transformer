from training import train_model, plot_model_comparisons

if __name__ == "__main__":
    uniform_results = train_model(p_method="uniform", embed_type="learned", patch_num=256, epochs_num=30)
    density_results = train_model(p_method="density", embed_type="coordinate", patch_num=202, epochs_num=30)

    plot_model_comparisons([uniform_results, density_results], metrics=('loss', 'accuracy'), save_path='uniform_vs_density.png')

    density_results_high = train_model(p_method="density", embed_type="coordinate", patch_num=202, epochs_num=20)
    density_results_mid = train_model(p_method="density", embed_type="coordinate", patch_num=160, epochs_num=20)
    density_results_low = train_model(p_method="density", embed_type="coordinate", patch_num=130, epochs_num=20)

    plot_model_comparisons([density_results_high, density_results_mid, density_results_low], metrics=('accuracy'), save_path='density_patch_num.png')