import numpy as np
import os
from pyteomics import mzxml

# Specify the directory containing the mzXML files
directory_path = "write your directory"
output_directory = "write your directory"

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Automatically generate a list of mzXML files in the directory
file_list = [f for f in os.listdir(directory_path) if f.endswith('.mzXML')]


def load_mzxml(filepath):
    peaks = []
    retention_times = []

    with mzxml.read(filepath) as reader:
        for spec in reader:
            mz_array = spec['m/z array']
            intensity_array = spec['intensity array']
            retention_time = spec['retentionTime']
            peaks.append(np.vstack((mz_array, intensity_array)).T)
            retention_times.append(retention_time)

    return peaks, np.array(retention_times)


def ROIpeaks(peaks, thresh, mzerror, minroi, time):
    nrows = len(peaks)
    mzroi = []
    MSroi = []
    roicell = []
    nmzroi = 0

    for irow in range(nrows):
        A = peaks[irow].astype(float)
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

    mzroi = np.array(sorted(mzroi))
    roicell = [roicell[i] for i in np.argsort(mzroi)]

    numberroi = [len(roicell[i][0]) for i in range(nmzroi)]
    maxroi = [max(roicell[i][2]) for i in range(nmzroi)]

    iroi = np.where((np.array(numberroi) > minroi) & (np.array(maxroi) > thresh))[0]

    mzroi = mzroi[iroi]
    nmzroi = len(mzroi)
    roicell = [roicell[i] for i in iroi]

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

    return mzroi, MSroi, roicell


# Parameters (to be provided by the user)
thresh = float(input("Enter the intensity threshold: "))
mzerror = float(input("Enter the m/z error tolerance: "))
minroi = int(input("Enter the minimum number of retention times to be considered in an ROI: "))

# Process each sample and save the matrix
for i, file_name in enumerate(file_list):
    filepath = os.path.join(directory_path, file_name)
    peaks, time = load_mzxml(filepath)

    mzroi, MSroi, roicell = ROIpeaks(peaks, thresh, mzerror, minroi, time)

    # Save the ROI matrix and mzroi vector for each file
    base_name = os.path.splitext(file_name)[0]
    output_matrix_file = os.path.join(output_directory, f"{base_name}_matrix.npy")
    output_mzroi_file = os.path.join(output_directory, f"{base_name}_mzroi.npy")

    np.save(output_matrix_file, MSroi)
    np.save(output_mzroi_file, mzroi)

    print(f"Saved ROI matrix for {file_name} as {output_matrix_file}")
    print(f"Saved m/z ROI vector for {file_name} as {output_mzroi_file}")

print("All matrices and m/z ROI vectors have been processed and saved successfully.")
