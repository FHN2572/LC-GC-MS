# Import necessary libraries
import numpy as np
import pandas as pd
import scipy.io as sio
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pyopenms
import matplotlib.pyplot as plt

# Define the path to the design matrix CSV file and MZXML file
design_file_path = "write your directory"
mzxml_file_path = "write your directory"

# Load the design matrix from the CSV file
design_df = pd.read_csv(design_file_path)

# Define the load_mzxml function to load the mzXML file
def load_mzxml(filepath):
    exp = pyopenms.MSExperiment()
    pyopenms.MzXMLFile().load(filepath, exp)

    peaks = []
    retention_times = []

    for spec in exp:
        mz_array, intensity_array = spec.get_peaks()
        peaks.append(np.vstack((mz_array, intensity_array)).T)
        retention_times.append(spec.getRT())

    return peaks, np.array(retention_times)

# Define the ROIpeaks function
def ROIpeaks(peaks, thresh, mzerror, minroi, time):
    nrows = len(peaks)
    print(f'Number of spectra (elution times) to process is: {nrows}')
    mzroi = []
    MSroi = []
    roicell = []
    nmzroi = 0

    for irow in range(nrows):
        print(f'Processing MS spectrum (elution time): {irow+1}')
        A = peaks[irow]
        A = A.astype(float)
        ipeak = np.where(A[:, 1] > thresh)[0]

        if ipeak.size > 0:
            mz = A[ipeak, 0]
            MS = A[ipeak, 1]
            if irow == 0:
                mzroi.append(mz[0])
                roicell.append([[mz[0]], [time[irow]], [MS[0]], [irow], mz[0]])

            for i in range(len(mz)):
                ieq = np.where(np.abs(np.array(mzroi) - mz[i]) <= mzerror)[0]

                if ieq.size > 0:
                    ieq = ieq[0]
                    roicell[ieq][0].append(mz[i])
                    roicell[ieq][1].append(time[irow])
                    roicell[ieq][2].append(MS[i])
                    roicell[ieq][3].append(irow)
                    roicell[ieq][4] = np.mean(roicell[ieq][0])
                    mzroi[ieq] = roicell[ieq][4]
                else:
                    nmzroi += 1
                    roicell.append([[mz[i]], [time[irow]], [MS[i]], [irow], mz[i]])
                    mzroi.append(mz[i])

    print(f'Initial number of ROI: {nmzroi}')
    mzroi = np.array(sorted(mzroi))

    roicell = [roicell[i] for i in np.argsort(mzroi)]

    numberroi = [len(roicell[i][0]) for i in range(nmzroi)]
    maxroi = [max(roicell[i][2]) for i in range(nmzroi)]

    iroi = np.where((np.array(numberroi) > minroi) & (np.array(maxroi) > thresh))[0]

    mzroi = mzroi[iroi]
    nmzroi = len(mzroi)
    roicell = [roicell[i] for i in iroi]

    print(f'Final number of ROI: {nmzroi}')

    MSroi = np.zeros((nrows, nmzroi))

    for i in range(nmzroi):
        nval = len(roicell[i][3])
        for j in range(nval):
            irow = roicell[i][3][j]
            MSI = roicell[i][2][j]
            MSroi[irow, i] += MSI

        y = MSroi[:, i]
        iy = np.where(y > 0)[0]
        if len(iy) > 1:
            intertime = time[iy[0]:iy[-1] + 1]
            ynew = np.interp(intertime, time[iy], y[iy])
            MSroi[iy[0]:iy[-1] + 1, i] = ynew

        MSroi[:, i] += np.random.randn(nrows) * 0.3 * thresh

    plt.figure()
    plt.bar(mzroi, np.sum(MSroi, axis=0), width=mzerror)
    plt.title('MS Spectra (ROI mz values)')

    plt.figure()
    plt.plot(time[:nrows], MSroi)
    plt.title('Chromatograms at ROI mz values')

    return mzroi, MSroi, roicell

# Loop through each parameter set
output_directory ="write your directory"  # Update this to your desired output directory
for i, params in design_df.iterrows():
    print(f"\nRunning ROI analysis with parameter set {i+1}/{len(design_df)}")
    print(f"Parameters: {params.to_dict()}")

    # Extract the parameters from the current row
    thresh = params['SignalIntensity']
    mzerror = params['MassError']
    minroi = params['Occurrences']

    # Process the sample with the current parameter set
    peaks, time = load_mzxml(mzxml_file_path)

    mzroi, MSroi, roicell = ROIpeaks(peaks, thresh, mzerror, minroi, time)

    # Save the results for this parameter set
    output_file = os.path.join(output_directory, f"sample_{i+1}_matrix.npy")
    np.save(output_file, MSroi)
    print(f"Saved ROI matrix for parameter set {i+1} as {output_file}")

print("All parameter sets have been processed and saved successfully.")
