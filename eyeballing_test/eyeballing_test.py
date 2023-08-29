import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation


v_source = 98.0 # km/s
fwhm = 8.5 # km/s
intensity_threshold = -6
frequency_error = 0.1 # MHz
peak_identification_cutoff_scaling = 0.5
peak_matching_radius_scaling = 0.7


PLOT_FULL = False
PLOT_EACH = True
#########################################################################


def rest_to_sky_freq(restfreq):
    return restfreq * (1.0 - v_source / 299792.458)

def sky_to_rest_freq(skyfreq):
    return skyfreq / (1.0 - v_source / 299792.458)

def dv_to_dnu(dv, nu0):
    return nu0 * dv / 299792.458


# load observed spectrum
with open("obs_spec.tsv", 'r') as file_obj:
    obs_freq, obs_I = np.loadtxt(file_obj, usecols=(0, 1), unpack=True)

obs_freq = obs_freq * 1000.0 # GHz to MHz
obs_freq_lower = np.min(obs_freq)
obs_freq_upper = np.max(obs_freq)
n_channel = len(obs_freq)
obs_I_smoothed = np.convolve(obs_I, np.array([0.25, 0.5, 0.25]), mode='same') # hanning smooth

mad = median_abs_deviation(obs_I_smoothed)

# find peaks
# peaks, _ = find_peaks(obs_I_smoothed, height=(peak_identification_cutoff_scaling*mad*1.4826))
peaks, _ = find_peaks(obs_I_smoothed, height=(mad*0.6), distance=3)

class Peak:
    def __init__(self, peak_index):
        self.sky_frequency = obs_freq[peak_index]
        self.amplitude = obs_I[peak_index]
        self.derived_rest_frequency = sky_to_rest_freq(obs_freq[peak_index])
        self.db_matching_radius = dv_to_dnu(fwhm, obs_freq[peak_index]) * peak_matching_radius_scaling
        self.transition_candidates_cdms = []
        self.matched_transitions_cdms = []
        self.not_matched_transitions_cdms = []
        self.uncertain_transitions_cdms = []
        self.transition_candidates_jpl = []
        self.matched_transitions_jpl = []
        self.not_matched_transitions_jpl = []
        self.uncertain_transitions_jpl = []

# create peak objects
observed_peaks = []
for peak_index in peaks:
    print("Emission peak at:", obs_freq[peak_index])
    observed_peaks.append(Peak(peak_index))


# load transition database cdms
with open("./combined_database_CDMS_patched.txt", 'r') as file_obj:
    line_db_cdms = file_obj.readlines()

# make a small version of the line_db_cdms to speed up
line_db_cdms_small = []
for transition in line_db_cdms:
    rest_frequency = float(transition[34:48])
    rest_frequency_error = float(transition[48:56])
    intensity_300k = float(transition[56:64])
    shifted_frequency = rest_to_sky_freq(rest_frequency)
    if shifted_frequency < obs_freq_upper and shifted_frequency > obs_freq_lower and rest_frequency_error < frequency_error and intensity_300k > intensity_threshold:
        line_db_cdms_small.append(transition[:-1])

# get a rest frequency array cdms for matching peaks
line_db_cdms_small_restfreq = np.zeros(len(line_db_cdms_small))
for id, transition in enumerate(line_db_cdms_small):
    rest_frequency = float(transition[34:48])
    line_db_cdms_small_restfreq[id] = rest_frequency

# load transition database jpl
with open("./combined_database_JPL.txt", 'r') as file_obj:
    line_db_jpl = file_obj.readlines()

# make a small version of the line_db_jpl to speed up
line_db_jpl_small = []
for transition in line_db_jpl:
    rest_frequency = float(transition[34:48])
    rest_frequency_error = float(transition[48:56])
    intensity_300k = float(transition[56:64])
    shifted_frequency = rest_to_sky_freq(rest_frequency)
    if shifted_frequency < obs_freq_upper and shifted_frequency > obs_freq_lower and rest_frequency_error < frequency_error and intensity_300k > intensity_threshold:
        line_db_jpl_small.append(transition[:-1])

# get a rest frequency array jpl for matching peaks
line_db_jpl_small_restfreq = np.zeros(len(line_db_jpl_small))
for id, transition in enumerate(line_db_jpl_small):
    rest_frequency = float(transition[34:48])
    line_db_jpl_small_restfreq[id] = rest_frequency




# matching cdms db transitions to observed peaks
for peak in observed_peaks:
    tmp_matched_id = np.where(np.abs(line_db_cdms_small_restfreq - peak.derived_rest_frequency) <= peak.db_matching_radius)
    #print(f"sky:{peak.sky_frequency}, derived_rest:{peak.derived_rest_frequency}, matching_radius:{peak.db_matching_radius}, {tmp_matched_id}")
    for id in tmp_matched_id:
        if len(id) != 0:
            for _id in id:
                peak.transition_candidates_cdms.append(line_db_cdms_small[_id])


#print("##############################################################################")


# matching jpl db transitions to observed peaks
for peak in observed_peaks:
    tmp_matched_id = np.where(np.abs(line_db_jpl_small_restfreq - peak.derived_rest_frequency) <= peak.db_matching_radius)
    #print(f"sky:{peak.sky_frequency}, derived_rest:{peak.derived_rest_frequency}, matching_radius:{peak.db_matching_radius}, {tmp_matched_id}")
    for id in tmp_matched_id:
        if len(id) != 0:
            for _id in id:
                peak.transition_candidates_jpl.append(line_db_jpl_small[_id])





# load the list of correctly classified species
correctly_classified_species = []
with open("./classified_correct.txt", 'r') as file_obj:
    tmp = file_obj.readlines()
    for ii in range(4, len(tmp)-1, 1):
        if tmp[ii][2] == " ":
            correctly_classified_species.append("0"+tmp[ii][3:8])
        else:
            correctly_classified_species.append(tmp[ii][2:8])

#print(correctly_classified_species)

# load the list of incorrectly classified species
incorrectly_classified_species = []
with open("./classified_incorrect.txt", 'r') as file_obj:
    tmp = file_obj.readlines()
    for ii in range(4, len(tmp)-1, 1):
        if tmp[ii][2] == " ":
            incorrectly_classified_species.append("0"+tmp[ii][3:8])
        else:
            incorrectly_classified_species.append(tmp[ii][2:8])

#print(incorrectly_classified_species)

# classify candidates based on CNN results
fobj = open("./log.txt", 'w')
for peak in observed_peaks:
    print(f"sky:{peak.sky_frequency}, derived_rest:{peak.derived_rest_frequency}, matching_radius:{peak.db_matching_radius}")
    fobj.write(f"sky:{peak.sky_frequency}, derived_rest:{peak.derived_rest_frequency}, matching_radius:{peak.db_matching_radius}\n")
    if len(peak.transition_candidates_cdms) != 0:
        for transition in peak.transition_candidates_cdms:
            mol_tag = transition[0:6]
            if mol_tag in correctly_classified_species:
                peak.matched_transitions_cdms.append(transition)
                print(f"        matched: {transition}")
                fobj.write(f"        matched: {transition}\n")
            elif mol_tag in incorrectly_classified_species:
                peak.not_matched_transitions_cdms.append(transition)
                print(f"    not matched: {transition}")
                fobj.write(f"    not matched: {transition}\n")
            else:
                peak.uncertain_transitions_cdms.append(transition)
                print(f"      uncertain: {transition}")
                fobj.write(f"      uncertain: {transition}\n")
fobj.close()

if PLOT_FULL == True:
    plt.figure(figsize=(16,9))
    plt.plot(obs_freq, obs_I, drawstyle="steps-mid")

    for peak in observed_peaks:
        plt.vlines(peak.sky_frequency, 0, np.max(obs_I), color='k', alpha=0.1)
        if len(peak.transition_candidates_jpl) != 0:
            for label in peak.transition_candidates_jpl:
                plt.vlines(rest_to_sky_freq(float(label[34:48])), 0, peak.amplitude*2, color='b')
                plt.text(rest_to_sky_freq(float(label[34:48])), 0, label[:76], rotation=90)

        if len(peak.transition_candidates_cdms) != 0:
            for label in peak.uncertain_transitions_cdms:
                plt.vlines(rest_to_sky_freq(float(label[34:48])), 0, peak.amplitude, color='g')
                plt.text(rest_to_sky_freq(float(label[34:48])), 0, label[:76], rotation=90)
            for label in peak.not_matched_transitions_cdms:
                plt.vlines(rest_to_sky_freq(float(label[34:48])), 0, peak.amplitude, color='r', alpha=0.33)
                plt.text(rest_to_sky_freq(float(label[34:48])), 0, label[:76], rotation=90, alpha=0.33)
            for label in peak.matched_transitions_cdms:
                plt.vlines(rest_to_sky_freq(float(label[34:48])), 0, peak.amplitude, color='r')
                plt.text(rest_to_sky_freq(float(label[34:48])), 0, label[:76], rotation=90)

    plt.show()



if PLOT_EACH == True:
    for target_tag in incorrectly_classified_species:
        if not target_tag == "032504":
            continue
        target_frequency = []
        target_label = []
        for transition in line_db_cdms_small:
            if transition[0:6] == target_tag:
                target_frequency.append(float(transition[34:48]))
                target_label.append(transition[:-1])

        plt.figure(figsize=(16,9))
        plt.plot(obs_freq, obs_I, drawstyle="steps-mid")
        for peak in observed_peaks:
            plt.vlines(peak.sky_frequency, 0, np.max(obs_I), color='k', alpha=0.1)

        # label all possible transitions from the cdms db
        for index, frequency in enumerate(target_frequency):
            plt.vlines(rest_to_sky_freq(frequency), 0, np.max(obs_I), color='g', alpha=0.33)
            plt.text(rest_to_sky_freq(frequency), 0, target_label[index][:76], rotation=90)

        for peak in observed_peaks:
            if len(peak.matched_transitions_cdms) != 0:
                for label in peak.matched_transitions_cdms:
                    if label[0:6] == target_tag:
                        plt.vlines(rest_to_sky_freq(float(label[34:48])), 0, np.max(obs_I), color='r')
                        plt.text(rest_to_sky_freq(float(label[34:48])), 0, label[:76], rotation=90)


        plt.show()

