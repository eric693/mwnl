def bler_to_ber(bler, bits_per_block):
    """
    Convert Block Error Rate (BLER) to Bit Error Rate (BER).

    :param bler: Block Error Rate
    :param bits_per_block: Number of bits per block
    :return: Bit Error Rate
    """
    ber = 1 - (1 - bler) ** (1 / bits_per_block)
    return ber

def ber_to_bler(ber, bits_per_block):
    """
    Convert Bit Error Rate (BER) to Block Error Rate (BLER).

    :param ber: Bit Error Rate
    :param bits_per_block: Number of bits per block
    :return: Block Error Rate
    """
    bler = 1 - (1 - ber) ** bits_per_block
    return bler

# Example usage
bler = 0.05




 # Example BLER value
bits_per_block = 20 # Example number of bits per block

# Convert BLER to BER
ber = bler_to_ber(bler, bits_per_block)
print(f"BER: {ber}")

# Convert BER back to BLER
bler_converted = ber_to_bler(ber, bits_per_block)
print(f"BLER (converted back): {bler_converted}")
