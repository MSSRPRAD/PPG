import numpy as np
from ppg.feature import extract_ppg45
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Generate random data
X = np.random.random((400000, 1250))

def process_row(index_row):
    """Process a single row with its index."""
    index, row = index_row
    return index, np.array(extract_ppg45(row, sample_rate=1250))

if __name__ == "__main__":
    # Define the number of processes
    num_processes = cpu_count()*2

    # Prepare indexed rows for processing
    indexed_data = list(enumerate(X))

    # Use multiprocessing to process rows while preserving order
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_row, indexed_data), total=len(indexed_data)))

    # Sort results by index to ensure order is preserved
    results.sort(key=lambda x: x[0])
    processed_data = np.array([result[1] for result in results])

    print("Processed data shape:", processed_data.shape)
