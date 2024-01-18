# import torch
# import numpy as np

# def generate_dci_bits(length):
#     # Generate a random DCI bit sequence
#     return torch.randint(0, 2, (length,), dtype=torch.int32)

# def transmit_through_awgn(dci_bits, snr_db):
#     # Convert SNR from dB to linear scale
#     snr_linear = 10**(snr_db / 10)
    
#     # Calculate the noise variance
#     noise_variance = 1 / (2 * snr_linear)
    
#     # Generate AWGN noise
#     noise = torch.randn(dci_bits.size()) * torch.sqrt(torch.tensor(noise_variance))
    
#     # Transmit DCI bits through AWGN channel
#     received_signal = dci_bits.float() + noise
    
#     return received_signal

# def receive_and_decode(received_signal, threshold=0.5):
#     # Decide on the bit value based on a threshold
#     decoded_bits = received_signal > threshold
#     return decoded_bits.int()

# def calculate_bler(original_bits, decoded_bits):
#     # Calculate the number of incorrect blocks
#     errors = torch.sum(original_bits != decoded_bits).item()
    
#     # BLER is the ratio of incorrect blocks to total blocks
#     bler = errors / original_bits.numel()
#     return bler

# # Define the length of the DCI and SNR
# dci_length = 100  # for example
# snr_db = 10  # signal to noise ratio in dB

# # Generate DCI bits
# dci_bits = generate_dci_bits(dci_length)

# # Transmit DCI through AWGN channel
# received_signal = transmit_through_awgn(dci_bits, snr_db)
# print(received_signal)

# # Receiver decodes the signal
# decoded_bits = receive_and_decode(received_signal)

# # Calculate BLER
# bler = calculate_bler(dci_bits, decoded_bits)

# print(f'DCI Bits: {dci_bits}')
# print(f'Decoded Bits: {decoded_bits}')
# print(f'BLER: {bler}')


# x = 1.8750
# n = 16
# z = x - 0.109375*n 
# print(z)

#===================

from scipy.special import erfc
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, randn

# Function to calculate BER for 16-QAM using simulation
# def simulate_16QAM_BER(EbNodB, n_bits):
#     M = 16  # For 16-QAM
#     k = int(np.log2(M))  # Bits per symbol
    
#     # Generating random bits
#     x = np.random.randint(0, 2, n_bits)
    
#     # Modulating the bits into 16-QAM symbols
#     symbols = np.reshape(x, (-1, k))  # Reshape into k-bits each
#     symbols_decimal = symbols.dot(2**np.arange(k)[::-1])  # Convert to decimal
#     y = np.exp(1j * (np.pi/M) * symbols_decimal)  # 16-QAM Modulation
    
#     # Transmitting symbols through AWGN channel
#     noise_variance = 1/(2 * (10**(EbNodB/10)))
#     noise = np.sqrt(noise_variance) * (randn(len(y)) + 1j * randn(len(y)))  # AWGN noise
#     y_noisy = y + noise  # Received signal
    
#     # Demodulating symbols back to bits
#     received_symbols_decimal = np.floor(((np.angle(y_noisy) % (2 * np.pi)) / (2 * np.pi)) * M).astype(int)
#     z = np.array([list(np.binary_repr(symbol, width=k)) for symbol in received_symbols_decimal], dtype=int).flatten()
    
#     # Calculating Bit Error Rate
#     errors = np.sum(x != z)
#     ber = errors / n_bits
#     return ber

# # Parameters
# n_bits = 10000  # Number of bits for simulation
# EbNodB_range = np.arange(-10, 11, 1)  # Range of Eb/N0 values from -10 to 10 dB

# # Simulate BER for 16-QAM
# # ber_simulated = [simulate_16QAM_BER(EbNodB, n_bits) for EbNodB in EbNodB_range]

# # Theoretical BER Calculation for 16-QAM
# M = 16
# k = np.log2(M)
# EbNo = 10**(EbNodB_range/10)
# x = np.sqrt(3*k*EbNo/(M-1))
# Pb_theoretical = (4/k) * (1 - 1/np.sqrt(M)) * (1/2) * erfc(x/np.sqrt(2))

# # Plotting the results
# plt.figure(figsize=(10, 7))
# plt.semilogy(EbNodB_range, Pb_theoretical, 'b-', label='Theoretical BER')
# # plt.semilogy(EbNodB_range, ber_simulated, 'ro', label='Simulated BER')
# plt.grid(True, which='both')
# plt.xlabel('Eb/N0 (dB)')
# plt.ylabel('Bit Error Rate (BER)')
# plt.title('BER vs Eb/N0 for 16-QAM in AWGN')
# plt.legend()
# plt.savefig("16-QAM")

# # Create a DataFrame from the SNR and BER values
# data = {
#     "SNR (dB)": EbNodB_range,
#     "BER": Pb_theoretical
# }

# df = pd.DataFrame(data)

# # Display the DataFrame as a table
# print(df)

#=========================

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import erfc
# import pandas as pd

# # Parameters for 64-QAM
# M = 256 # Modulation order for 64-QAM
# k = np.log2(M)  # Number of bits per symbol for 64-QAM
# EbNodB_range = np.arange(-10, 18, 1)  # Range of Eb/N0 values from -10 to 10 dB

# # Theoretical BER Calculation for 64-QAM
# EbNo = 10**(EbNodB_range/10)
# x = np.sqrt(3*k*EbNo/(M-1))
# Pb_theoretical_64QAM = (4/k) * (1 - 1/np.sqrt(M)) * (1/2) * erfc(x/np.sqrt(2))

# # Plotting the results for 64-QAM
# plt.figure(figsize=(10, 7))
# plt.semilogy(EbNodB_range, Pb_theoretical_64QAM, 'b-', label='Theoretical BER for 64-QAM')
# plt.grid(True, which='both')
# plt.xlabel('Eb/N0 (dB)')
# plt.ylabel('Bit Error Rate (BER)')
# plt.title('BER vs Eb/N0 for 64-QAM in AWGN')
# plt.legend()
# # Save the plot if running locally
# # plt.savefig("64-QAM_BER_Theoretical.png")
# plt.show()

# # Create a DataFrame from the SNR and BER values for 64-QAM
# data_64QAM = {
#     "SNR (dB)": EbNodB_range,
#     "BER": Pb_theoretical_64QAM
# }

# # Creating a DataFrame and displaying it
# df_64QAM = pd.DataFrame(data_64QAM)
# print(df_64QAM)


# Initialize the starting value and the step size
# start_value = 0.375
# end_value = 1.625
# step_size = (end_value - start_value) / 63

# # Create a for loop to iterate and print each step
# for i in range(64):
#     current_value = start_value + i * step_size
#     print(current_value)




# To calculate the exponential distribution within the range 0.375 and 1.625
# and divide this range into 16 equal parts, we can use numpy's linspace to find the edges
# and then calculate the exponential of these edges to maintain the non-uniform distribution.

# Define the range and number of blocks
# start, end = 0.3750, 1.625
# blocks = 15

# # Evenly spaced values in the exponential space
# exp_space_values = np.logspace(np.log10(start), np.log10(end), blocks+1)

# # The resulting values represent the edges of 16 blocks in exponential distribution
# print(exp_space_values)


# ## power law distribution ##
# start, end = 0.3750, 1.625
# divisions = 15

# # Generate the power law distribution
# power = 10  # Higher powers will make the distribution more extreme
# x = np.linspace(0, 1, divisions + 1)**power
# x = x * (end - start) / max(x)  # Normalize to the range of interest
# extreme_values = start + x  # Shift to start from 0.375

# print(extreme_values)

## Gaussian mixture distributions
# import numpy as np

# # Define the parameters for two Gaussian distributions
# mean1, std_dev1 = 1.875, 0.05
# mean2, std_dev2 = 0.125, 0.05

# # Generate samples for each distribution
# samples1 = np.random.normal(mean1, std_dev1, 8)
# samples2 = np.random.normal(mean2, std_dev2, 8)

# # Combine the samples to form a bimodal distribution
# bimodal_samples = np.concatenate([samples1, samples2])

# print(bimodal_samples)

# To classify 6-bit binary numbers based on the number of '1's they contain
# We will create two groups: one for numbers with 0, 1, or 2 ones (Group A)
# and another for numbers with 3, 4, 5, or 6 ones (Group B).

# Define the groups
# group_A = []  # For 0, 1, or 2 ones
# group_B = []  # For 3, 4, 5, or 6 ones

# # Check all 6-bit numbers
# for i in range(16):  # 64 possibilities for 6-bit numbers (2^6)
#     binary_num = format(i, '04b')  # Get binary representation
#     count_ones = binary_num.count('1')  # Count the number of ones

#     # Assign to groups based on the count of ones
#     if count_ones in [0, 1]:
#         group_A.append(binary_num)
#     else:
#         group_B.append(binary_num)

# print("group A :")
# print(group_A)
# print("group B :")
# print(group_B)

import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
import pandas as pd

# Parameters for QPSK
M_QPSK = 4
k_QPSK = np.log2(M_QPSK)

# Range of Eb/N0 values from -10 to 17 dB
EbNodB_range_QPSK = np.arange(-10, 11, 1)

# Theoretical BER Calculation for QPSK
EbNo_QPSK = 10**(EbNodB_range_QPSK / 10)
x_QPSK = np.sqrt(2 * k_QPSK * EbNo_QPSK)
Pb_theoretical_QPSK = (1/2) * erfc(x_QPSK / np.sqrt(2))

# Plotting the results for QPSK
plt.figure(figsize=(10, 7))
plt.semilogy(EbNodB_range_QPSK, Pb_theoretical_QPSK, 'r-', label='Theoretical BER for QPSK')
plt.grid(True, which='both')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs Eb/N0 for QPSK in AWGN')
plt.legend()

# Displaying the plot
plt.show()

# Create a DataFrame from the SNR and BER values for QPSK
data_QPSK = {
    "SNR (dB)": EbNodB_range_QPSK,
    "BER": Pb_theoretical_QPSK
}

# Creating a DataFrame and displaying it
df_QPSK = pd.DataFrame(data_QPSK)
# df_QPSK.head()  # Display the first few rows of the DataFrame
# print(df_QPSK)
# Converting BER values to non-exponential format for better readability
df_QPSK_non_exp = df_QPSK.copy()
df_QPSK_non_exp['BER'] = df_QPSK_non_exp['BER'].apply(lambda x: '{:.10f}'.format(x))

df_QPSK_non_exp.head()  # Display the first few rows of the DataFrame with non-exponential BER values
print(df_QPSK_non_exp)
