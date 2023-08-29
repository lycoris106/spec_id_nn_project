#!/bin/sh

python3 ./gauss_fit.py \
    --input_file "mock_spec_mal.txt" \
    --rad_v 0.0 \
    --fwhm_guess 2.5 \
    --freq_unit "MHz" \
    --y_unit "K" \
    --bmaj 3.446816936134E-04 \
    --bmin 2.289818422717E-04 \
    --istxt \
    # --do_plot \

python3 ./database_match.py \
    --input_file "mock_spec_mal" \
    --print_match \
    --freq_unit "MHz" \
    --y_unit "K" \
    --bmaj 3.446816936134E-04 \
    --bmin 2.289818422717E-04 \
    --istxt \
    # --do_plot \

python3 ./model_infer.py \
    --input_file "mock_spec_mal" \

python3 result_analysis.py \
    --input_file "mock_spec_mal" \
    --freq_unit "MHz" \
    --rad_v 0.0 \
    --plot_all \
    --bmaj 3.446816936134E-04 \
    --bmin 2.289818422717E-04 \
    --istxt \


