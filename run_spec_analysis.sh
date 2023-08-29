#!/bin/sh

python3 ./gauss_fit.py \
    --input_file "member.uid___A001_X1284_X6c1._G31.41p0.31__sci.spw25.cube.I.pbcor_subimage.fits-Z-profile-Region_1-Statistic_Mean-Cooridnate_Current-2023-07-11-10-21-17.tsv" \
    --rad_v 98.0 \
    --fwhm_guess 4.0 \
    --freq_unit "GHz" \
    --y_unit "Jy/beam" \
    --do_plot \
    --bmaj 3.446816936134E-04 \
    --bmin 2.289818422717E-04 \
    # --istxt \

python3 ./database_match.py \
    --input_file "member.uid___A001_X1284_X6c1._G31.41p0.31__sci.spw25.cube.I.pbcor_subimage.fits-Z-profile-Region_1-Statistic_Mean-Cooridnate_Current-2023-07-11-10-21-17" \
    --print_match \
    --freq_unit "GHz" \
    --y_unit "Jy/beam" \
    --bmaj 3.446816936134E-04 \
    --bmin 2.289818422717E-04 \
    # --do_plot \
    # --istxt \

python3 ./model_infer.py \
    --input_file "member.uid___A001_X1284_X6c1._G31.41p0.31__sci.spw25.cube.I.pbcor_subimage.fits-Z-profile-Region_1-Statistic_Mean-Cooridnate_Current-2023-07-11-10-21-17" \

python3 result_analysis.py \
    --input_file "member.uid___A001_X1284_X6c1._G31.41p0.31__sci.spw25.cube.I.pbcor_subimage.fits-Z-profile-Region_1-Statistic_Mean-Cooridnate_Current-2023-07-11-10-21-17" \
    --freq_unit "GHz" \
    --rad_v 98.0 \
    --plot_all \
    --bmaj 3.446816936134E-04 \
    --bmin 2.289818422717E-04 \
    # --istxt \


