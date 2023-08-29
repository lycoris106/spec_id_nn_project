import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd
import csv
import argparse
import os

SPEED_OF_LIGHT = 299792.458
MODEL_FREQ_MAX = 85.0 # excluded
MODEL_FREQ_MIN = 84.0
MODEL_N_CHANNELS = 2000
MODEL_CHANNEL_WIDTH=(MODEL_FREQ_MAX - MODEL_FREQ_MIN) / MODEL_N_CHANNELS

class GaussFit:
    """
    Input: The original tsv file of a spectrum
    Output: A tsv file of gaussians that fit the original spectrum
    """

    def __init__(self, input_file, radial_v, freq_unit, y_unit, fwhm_guess=4.0, istxt=False, bmaj=3.446816936134E-04, bmin=2.289818422717E-04, smooth_wind_size=7, peak_wind_width=2):
        self.input_file = os.path.splitext(input_file)[0]
        if istxt:
            self.spec_df = pd.read_csv(f"Spectra/{input_file}", delimiter='\t', skiprows=1, usecols=[0, 2], names=['freq', 'value'])
        else:
            self.spec_df = pd.read_csv(f"Spectra/{input_file}", sep='\t', comment='#', header=None, names=['freq', 'value'])
        if freq_unit == "MHz":
            self.spec_df[['freq']] = self.spec_df[['freq']] / 1000
        self.radial_v = radial_v
        # self.spec_df[['freq']] = self.spec_df[['freq']] / (1 + radial_v / SPEED_OF_LIGHT)
        self.n_channels = self.spec_df.shape[0] #1918
        self.fwhm_guess = fwhm_guess
        self.freq_min = self.spec_df.loc[0, 'freq']
        self.freq_max = self.spec_df.loc[self.n_channels-1, 'freq']
        self.y_unit = y_unit
        self.bmaj, self.bmin = bmaj*3600, bmin*3600
        self.freq = self.spec_df["freq"].to_numpy()
        self.value = self.spec_df["value"].to_numpy()
        self.smooth_wind_size = smooth_wind_size
        self.peak_wind_width = peak_wind_width



        # self.fwhm_guess = [fwhm_guess]
        # self.maxfev = maxfev



    def freq2chan(self, num):
        res = (num - self.freq_min) * (self.n_channels-1) / (self.freq_max - self.freq_min)
        rounded_res = [int(round(num)) for num in res]
        # print(rounded_res)
        return rounded_res

    def chan2freq(self, num):
        # return num * (self.freq_max - self.freq_min) / (self.n_channels-1) + self.freq_min
        return self.freq[num]


    def _sum_gaussian(self, x, fwhms, means, amps):
        y = np.zeros_like(x)
        for i in range(0, len(fwhms)):
            y = y + amps[i] * np.exp(-4. * np.log(2) * (x-means[i])**2 / fwhms[i]**2)
        return y

    def _sum_gaussian_fix(self, x, fwhm, means, amps):
        y = np.zeros_like(x)
        for i in range(0, len(means)):
            y = y + amps[i] * np.exp(-4. * np.log(2) * (x-means[i])**2 / fwhm**2)
        return y


    def hanning_smooth(self, seq):
        # han = np.hanning(self.smooth_wind_size)
        han = np.array([0.25, 0.5, 0.25])
        y = np.convolve(seq, han/han.sum(), mode="same")
        # print(sum(y))
        return y

    def get_peaks_wind(self, y):
        mad = np.median(np.absolute(y - np.mean(y)))
        print(f"mad: {mad}")
        x_peaks, _ = find_peaks(y, height=mad*1.2, prominence=None)
        amps = [y[p] for p in x_peaks]
        return x_peaks, amps
        # x_peaks, amps = [], []
        # for (i, val) in enumerate(y):
        #     isPeak = True
        #     if val < self.peak_thres:
        #         continue
        #     if i >= self.peak_wind_width and i < self.n_channels-self.peak_wind_width:
        #         for j in range(1, self.peak_wind_width+1):
        #             if y[i-j] >= y[i-j+1]:
        #                 isPeak = False
        #                 break
        #         for j in range(1, self.peak_wind_width+1):
        #             if y[i+j] >= y[i+j-1]:
        #                 isPeak = False
        #                 break
        #         if isPeak:
        #             x_peaks.append(i)
        #             amps.append(y[i])
        # return x_peaks, amps

    def check_unit_and_convert(self):
        assumed_unit = "K"
        if self.y_unit != assumed_unit:
            print(f"Converting unit from {self.y_unit} to {assumed_unit}...")
            # Convert the frequency values to the assumed unit
            if self.y_unit == "Jy/beam":
                self.value = list(map(lambda x, y: y*1.22*1e6 / (x*x*self.bmaj*self.bmin), self.freq, self.value))

    def fit(self):
        self.check_unit_and_convert()
        self.x = self.freq2chan(self.freq)
        self.y = self.hanning_smooth(self.value)
        # self.y = self.value
        self.x_peaks, self.amps = self.get_peaks_wind(self.y)

        guess = []
        # for i in range(len(self.x_peaks)):
        #     guess.append(self.fwhm_guess)

        def helper(x, *fwhms):
            return self._sum_gaussian(x, fwhms, self.x_peaks, self.amps)

        def helper_fix(x, fwhm):
            return self._sum_gaussian_fix(x, fwhm, self.x_peaks, self.amps)

        # self.popt, _ = curve_fit(helper, self.x, self.y, p0=guess, maxfev=self.maxfev)
        # for p in self.x_peaks:
        #     print(self.freq[p])
        # self.popt, _ = curve_fit(helper_fix, self.x, self.y, p0=self.fwhm_guess, maxfev=self.maxfev, bounds=((0,), (np.inf,)))
        # self.popt = self.fixed_fwhm

        # print(self.popt)
        # self.fit_sum = self._sum_gaussian_fix(self.x, self.popt[0], self.x_peaks, self.amps)


    '''
    Methods for plotting
    '''

    def draw_gauss(self, mu, fwhm, amp):
        x = np.linspace(mu - 3.*fwhm, mu + 3.*fwhm, 100)
        self.ax.plot(x, amp * np.exp(-4. * np.log(2) * (x-mu)**2 / fwhm**2), color="blue")

    def plot_fit_result(self):
        fig, self.ax = plt.subplots()
        self.ax.plot(self.x, self.y, color="orange")
        # for i in range(len(self.x_peaks)):
        #     # print(amps[i])
        #     # self.draw_gauss(self.x_peaks[i], self.popt[i], self.amps[i])
        #     self.draw_gauss(self.x_peaks[i], self.fwhm_guess, self.amps[i])
        plt.show()

    """
    Methods for saving the result
    """

    def save_tsv(self):
        """
        save
        # freq_min, freq_max, n_channel
        1. the centers (in GHz)
        2. the amplitudes (in K)
        3. the fwhms (in channel)
        of the gaussians to a tsv
        """

        out_prop_filename = f"temp_files/{self.input_file}_gauss_param_v0.csv"
        with open(out_prop_filename, 'w') as output_file:
            writer = csv.writer(output_file, delimiter='\t')
            fwhm_0 = self.fwhm_guess*(self.freq_max-self.freq_min)/(self.n_channels-1) / MODEL_CHANNEL_WIDTH
            header_info = f"# freq_min:{self.freq_min}, freq_max:{self.freq_max}, n_channels:{self.n_channels}, radial_v:{self.radial_v}, fwhm_guess:{fwhm_0}"
            writer.writerow([header_info])
            writer.writerow(["mean", "amplitude"])
            # print(self.popt)
            for c, a in zip(map(self.chan2freq, self.x_peaks), self.amps):
                c_v0 = c / (1 - self.radial_v / SPEED_OF_LIGHT)
                writer.writerow((c_v0, a))
            # writer.writerows(zip(map(self.chan2freq, self.x_peaks), self.amps, self.popt))

        print(f"Parameters of matched gaussians have been written in {out_prop_filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for finding peaks in a spectrum and outputting the gaussian parameters')

    parser.add_argument('--input_file', type=str, help='name of input spectrum file')
    parser.add_argument('--rad_v', type=float, help='radial velocity relative to the observer')
    parser.add_argument('--freq_unit', choices=["GHz", "MHz"], help="unit for frequency: 'GHz' or 'MHz'")
    parser.add_argument('--y_unit', choices=["K", "Jy/beam"], help="unit for the y/intensity value: 'K' or 'Jy/beam'")
    parser.add_argument('--fwhm_guess', type=float, help="fixed guess of fwhm, *respect to the original channel width*")
    parser.add_argument('--istxt', action='store_true', help='is the file in txt format instead of csv/tsv')
    parser.add_argument('--do_plot', action='store_true', help='whether to plot the fitting result')
    parser.add_argument('--bmaj', type=float, default=3.446816936134E-04, help='bmaj, should be provided if y_unit == Jy/beam')
    parser.add_argument('--bmin', type=float, default=2.289818422717E-04, help='bmin, should be provided if y_unit == Jy/beam')


    args = parser.parse_args()

    # gauss_fit = GaussFit('member.uid___A001_X1284_X6c1._G31.41p0.31__sci.spw25.cube.I.pbcor_subimage.fits-Z-profile-Region_1-Statistic_Mean-Cooridnate_Current-2023-07-11-10-21-17.tsv', file_unit="Jy/beam")
    gauss_fit = GaussFit(input_file=args.input_file, radial_v=args.rad_v, freq_unit=args.freq_unit, y_unit=args.y_unit, fwhm_guess=args.fwhm_guess, istxt=args.istxt, bmaj=args.bmaj, bmin=args.bmin)
    gauss_fit.fit()
    if args.do_plot:
        gauss_fit.plot_fit_result()
    gauss_fit.save_tsv()
    print(f'freq_max: {gauss_fit.freq_max}')
    print(f'freq_min: {gauss_fit.freq_min}')



