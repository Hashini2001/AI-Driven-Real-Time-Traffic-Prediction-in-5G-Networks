import pandas as pd
import re
import os

# Load CSV
CSV_FILE = 'C:/Users/kdhas/Desktop/Research New/Dataset/drop_dataset/new_Zoom_2.csv'
df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

# Lowercase Info and Protocol for searching
df['Info_lower'] = df['Info'].astype(str).str.lower()
df['Protocol_lower'] = df['Protocol'].astype(str).str.lower()

# Extract flags as integers (1 or 0)
df['is_syn'] = df['Info_lower'].str.contains(r'\bsyn\b', regex=True).astype(int)
df['is_ack'] = df['Info_lower'].str.contains(r'\back\b', regex=True).astype(int)
df['is_fin'] = df['Info_lower'].str.contains(r'\bfin\b', regex=True).astype(int)
df['is_tls_handshake'] = (
    df['Info_lower'].str.contains('client hello') | 
    df['Info_lower'].str.contains('server hello')
).astype(int)
df['is_http'] = (df['Protocol_lower'] == 'http').astype(int)
df['is_udp'] = (df['Protocol_lower'] == 'udp').astype(int)
df['is_dns'] = (df['Protocol_lower'] == 'dns').astype(int)
df['is_db_lsp'] = df['Info_lower'].str.contains('db-lsp', regex=False).astype(int)
df['is_stun'] = df['Protocol_lower'].str.contains(r'\bstun\b').astype(int)
df['is_classic_stun'] = df['Info_lower'].str.contains(r'classic[\s_-]?stun', regex=True).astype(int)
df['is_rtcp'] = (df['Protocol_lower'] == 'rtcp').astype(int)
df['is_raknet'] = df['Protocol_lower'].str.contains('raknet', regex=False).astype(int)
df['is_quic'] = df['Protocol_lower'].str.fullmatch('quic').astype(int)
df['is_gquic'] = df['Protocol_lower'].str.fullmatch('gquic').astype(int)

# Extract source and destination ports from Info column
def extract_ports(info):
    match = re.search(r'(\d+)\s*>\s*(\d+)', info)
    if match:
        return match.group(1), match.group(2)
    else:
        return 'None', 'None'

df[['src_port', 'dst_port']] = df['Info'].apply(lambda x: pd.Series(extract_ports(str(x))))

# --- Packet rate and Byte rate calculations ---

# Convert Time to seconds (assume hh:mm:ss or mm:ss)
df['Time_sec'] = pd.to_timedelta(df['Time']).dt.total_seconds().astype(int)

df['type'] = 'Video_Conferencing'
# Group by Time_sec
rate_df = df.groupby('Time_sec').agg(
    packet_rate=('Length', 'count'),
    byte_rate=('Length', 'sum')
).reset_index()

# Merge packet/byte rate back to original DataFrame
df = df.merge(rate_df, on='Time_sec', how='left')

# Protocol-specific extraction
# UDP Length (if Length > 0 and protocol is UDP)
df['udp_length'] = df.apply(lambda row: int(row['Length']) if row['is_udp'] == 1 else 0, axis=1)

# DNS
df['dns_query_type'] = df['Info_lower'].str.extract(r'standard query [0-9xa-f]* (\w+)').fillna('None')
df['dns_query_domain'] = df['Info_lower'].str.extract(r'standard query [0-9xa-f]* \w+ ([\w\.-]+)').fillna('None')

# STUN
df['stun_response_type'] = df['Info_lower'].str.extract(r'(binding request|binding success response)').fillna('None')
df['mapped_ip'] = df['Info_lower'].str.extract(r'xor-mapped-address:\s*([\d\.]+)').fillna('None')
df['mapped_port'] = df['Info_lower'].str.extract(r'xor-mapped-address:\s*[\d\.]+:(\d+)').fillna('None')

# RTCP
df['rtcp_type'] = df['Info_lower'].str.extract(r'(sender report|receiver report|source description|goodbye|application-defined)').fillna('None')
df['rtcp_malformed'] = df['Info_lower'].str.contains('malformed packet').astype(int)

# RakNet
df['raknet_msg_type'] = df['Info_lower'].str.extract(r'(open connection reply \d+)').fillna('None')
df['raknet_malformed'] = df['Info_lower'].str.contains('malformed packet').astype(int)

# QUIC
df['is_quic_handshake'] = df['Info_lower'].str.contains('handshake').astype(int)
df['quic_scid'] = df['Info_lower'].str.extract(r'scid=([0-9a-f]+)').fillna('None')

# GQUIC
df['gquic_cid'] = df['Info_lower'].str.extract(r'cid:\s*([\d]+)').fillna('None')
df['gquic_packet_number'] = df['Info_lower'].str.extract(r'pkn:\s*([\d]+)').fillna('None')

# Select and order final columns
final_cols = ['Time', 'Source', 'Destination', 'Protocol', 'Length', 
              'is_syn', 'is_ack', 'is_fin', 'is_tls_handshake', 'is_http',
              'is_udp', 'is_dns', 'is_db_lsp', 'is_stun', 'is_classic_stun',
              'is_rtcp', 'is_raknet', 'is_quic', 'is_gquic', 
              'src_port', 'dst_port', 'packet_rate', 'byte_rate', 'type',
              'udp_length', 'dns_query_type', 'dns_query_domain',
              'stun_response_type', 'mapped_ip', 'mapped_port',
              'rtcp_type', 'rtcp_malformed',
              'raknet_msg_type', 'raknet_malformed',
              'is_quic_handshake', 'quic_scid',
              'gquic_cid', 'gquic_packet_number']

# Show the result
print(df[final_cols].head(5))

#Save to output file
output_folder = 'C:/Users/kdhas/Desktop/Research New/Dataset/preprocess_dataset'
output_file = os.path.join(output_folder, 'preprocessed_Zoom_2.csv')
df[final_cols].to_csv(output_file, index=False)

print(f"\n Preprocessed file saved to: {output_file}")
