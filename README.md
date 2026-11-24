# Calcium-gated model of tricellular zipping

Code for the simulations in Fig. 6I–K of

Adhikary B, Zhao XO, Shahi A, Laboy-Figueroa AG, Purvis MP, Tjoelker KK, Huycke TR, Banerjee S, Martin AC, Miller AL. *Calcium and actomyosin pulses drive tricellular zipping to remodel epithelial cell-cell junctions and facilitate cell packing.*

## What the code does

`zipping_sim.py` simulates the response of the bicellular junction to a single induced calcium flash, using the calcium-gated model described in the Supplementary Information. A calcium flash drives F-actin activity, which combines with a steady Myosin II level to set contractile activity. Contractile activity sets clutch engagement through a saturating switch, and the transmitted net force sets the junction elongation rate. The equations are integrated from 2 min before to 2 min after the flash. Elongation slopes before and after the flash are obtained from least-squares linear fits over these 2-min windows, as for the experimental data.

The script produces three figures:

- **6I**: response to a single reference flash (calcium, F-actin, clutch engagement, junction length).
- **6J**: pre- and post-flash slopes for ten flashes. Amplitude A is drawn uniformly from 0.6 to 1.5, slow decay time τ_slow from 0.5 to 3 min, baseline F-actin f₀ from 0.9 to 1.1, and Gaussian noise with standard deviation 0.03 µm is added to the length. The random seed is fixed, so the panel is reproducible.
- **6K**: increase in elongation slope as a function of τ_slow for A = 0.5, 1.0 and 1.5.

Parameter values are listed in the script and in Table S1 of the Supplementary Information.

## Requirements

Python 3.8 or later, with NumPy and Matplotlib. Tested with NumPy 2.4 to 2.5 and Matplotlib 3.10 to 3.11.

```
pip install numpy matplotlib
```

## Usage

```
python3 zipping_sim.py
```

Runtime is a few seconds. The script writes each figure as PDF, SVG and PNG to the current directory. It also prints a check line:

```
reference flash: s_pre = 0.136, s_post = 0.443 um/min | ensemble mean s_pre = 0.15, s_post = 0.38 | F at baseline = 0.93 (> 0, rectifier inactive)
```

## License

MIT
