import torch
import torch.nn as nn
import torch.optim as optim
from Input_Tokenizer import Tokenizer
from embedding_new import PacketEmbedding, FlowEmbedding
from packet_encoder import PacketLevelEncoder
from flow_encoder import FlowLevelEncoder
import os
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
from field_header_pos_encoding import field_pos, header_pos

# Hyperparameters
vocab_size = 262
embed_dim = 768
num_heads = 12
num_layers = 6
dropout = 0.1
max_flow_length = 512
mask_prob = 0.15
num_epochs = 1
max_len = 512
batch_size_1 = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# loading vocabulary
vocab = {}
custom_vocab_path = r"/home/satvik/spark/data/vocab_1.txt"
with open(custom_vocab_path, 'r', encoding='utf-8') as f:
    for line in f:
        token, token_id = line.strip().split('\t')
        vocab[token] = int(token_id)

# initialise the tokenizer
tokenizer = Tokenizer(vocab_file=custom_vocab_path)

# Initialize packet embedding and encoding modules
packet_embedding = PacketEmbedding(
    vocab_size, max_len=578 , embed_dim=embed_dim, dropout=dropout).to(device)
packet_encoder = PacketLevelEncoder(
    vocab_size, embed_dim, max_len, num_heads, num_layers, dropout).to(device)

# Initialize flow embedding and encoding module
flow_embedding = FlowEmbedding(embed_dim, max_flow_length, dropout, vocab).to(device)
flow_encoder = FlowLevelEncoder(
    embed_dim, num_layers, num_heads, dropout, vocab, max_flow_length, mask_prob)

# training criterion initialize
optimizer = optim.Adam(list(packet_embedding.parameters()) + list(flow_embedding.parameters()) + list(packet_encoder.parameters()) + list(flow_encoder.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

from torch.utils.data import Dataset, DataLoader

class PacketSequenceDataset(Dataset):
    def __init__(self, packet_seq_dir, field_pos_dir, header_pos_dir, tokenizer, batch_size_1):
        self.packet_seq_files = sorted([os.path.join(packet_seq_dir, file) for file in os.listdir(packet_seq_dir) if file.endswith('.txt')])
        self.field_pos_files = sorted([os.path.join(field_pos_dir, file) for file in os.listdir(field_pos_dir) if file.endswith('.txt')])
        self.header_pos_files = sorted([os.path.join(header_pos_dir, file) for file in os.listdir(header_pos_dir) if file.endswith('.txt')])
        self.tokenizer = tokenizer
        self.batch_size_1 = batch_size_1

        # Calculate total length by summing up chunks across files
        self.total_chunks = []
        for file in self.packet_seq_files:
            num_lines = len(open(file, 'r').readlines())
            num_chunks = (num_lines + self.batch_size_1 - 1) // self.batch_size_1
            self.total_chunks.append(num_chunks)
        self.total_len = sum(self.total_chunks)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Find which file and chunk the index maps to
        cumulative_chunks = 0
        for file_idx, num_chunks in enumerate(self.total_chunks):
            if cumulative_chunks + num_chunks > idx:
                line_idx = idx - cumulative_chunks
                break
            cumulative_chunks += num_chunks
        else:
            raise IndexError("Index out of range")

        packet_seq_file = self.packet_seq_files[file_idx]
        field_pos_file = self.field_pos_files[file_idx]
        header_pos_file = self.header_pos_files[file_idx]

        with open(packet_seq_file, 'r', encoding='utf-8') as f:
            hex_dumps = f.readlines()
        
        padded_all_tokens, token_ids, mask, max_length = self.tokenizer.encode_packet(hex_dumps)
        
        # Slice out the chunk from token_ids
        chunk_start = line_idx * self.batch_size_1
        chunk_end = min((line_idx + 1) * self.batch_size_1, token_ids.size(0))
        chunk = token_ids[chunk_start:chunk_end]
        
        field_posn = field_pos(field_pos_file, chunk_start, chunk_end)
        header_posn = header_pos(header_pos_file, chunk_start, chunk_end)
        
        return chunk, field_posn, header_posn, packet_seq_file


# Create the dataset and data loader without shuffling
dataset = PacketSequenceDataset(
    packet_seq_dir='/home/satvik/spark/spark2/packets', 
    field_pos_dir='/home/satvik/spark/spark2/fields', 
    header_pos_dir='/home/satvik/spark/spark2/headers', 
    tokenizer=tokenizer, 
    batch_size_1=batch_size_1
)
print("1")
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Check contents of train_loader
# for i, (packet_sequences, field_position, header_position, file_name) in enumerate(train_loader):
#     print(f"Batch {i+1}")
#     print("Packet Sequences:", packet_sequences.shape)
#     print("Field Position:", field_position.shape)
#     print("Header Position:", header_position.shape)
#     print("File Name:", file_name) 

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler('cuda')

direction_dir = "/home/satvik/spark/spark2/direction"
print("2")
def process_encodings(encodings, packet_number):
    """ Process accumulated encodings and load corresponding direction data. """
    final_packet_encodings = torch.cat(encodings, dim=0).to(device)
    print("Final concatenated shape:", final_packet_encodings.shape)

    # Load the corresponding direction data from the text file
    direction_file_name = f'direction_{packet_number}.txt'
    direction_file_path = os.path.join(direction_dir, direction_file_name)

    with open(direction_file_path, 'r') as file:
        direction_data = [int(line.strip()) for line in file.readlines()]

    # Convert direction data to a tensor
    direction_tensor = torch.tensor(direction_data, device=device)
    print(final_packet_encodings.device, direction_tensor.device)
    # Call FlowEmbedding with the accumulated packet encodings and direction data
    flow_embeddings, pad_indices = flow_embedding(final_packet_encodings, direction_tensor)
    print("Flow embeddings computed for packet:", packet_number)
    print("flow embeddings: ", flow_embeddings.shape)
    print("-------------------------------------------------------------")

    # Compute flow encoding and MPM loss
    flow_encoding, mpm_loss = flow_encoder(flow_embeddings, pad_indices)
    print("flow encoding: ", flow_encoding.shape)
    print("mpm loss: ", mpm_loss)
    print(type(mpm_loss))

    # Ensure mpm_loss is a tensor and on the same device
    mpm_loss_tensor = mpm_loss[0].to(device)
    print(mpm_loss_tensor)
    return mpm_loss_tensor

def save_model(save_path, epoch, model_components, optimizer, mlm_loss_load, sfbo_loss_load, mpm_loss_load):
    """
    Save model state and optimizer state to a file.

    Args:
        save_path (str): Filepath to save the model.
        epoch (int): Current epoch number.
        model_components (dict): Dictionary containing model state_dicts.
        optimizer (torch.optim): Optimizer object.
        loss (float): Final loss value.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_components,
        'optimizer_state_dict': optimizer.state_dict(),
        'mlm_loss': mlm_loss_load,
        'sfbo_loss': sfbo_loss_load,
        'mpm_loss': mpm_loss_load
    }, save_path)
    print(f"Model saved at {save_path}")
    
def load_model(load_path):
    """
    Load model state and optimizer state from a file.

    Args:
        load_path (str): Filepath to load the model.

    Returns:
        dict: Dictionary containing reinitialized model components and optimizer.
    """
    checkpoint = torch.load(load_path, map_location=device)
    packet_embedding.load_state_dict(checkpoint['model_state_dict']['packet_embedding'])
    packet_encoder.load_state_dict(checkpoint['model_state_dict']['packet_encoder'])
    flow_embedding.load_state_dict(checkpoint['model_state_dict']['flow_embedding'])
    flow_encoder.load_state_dict(checkpoint['model_state_dict']['flow_encoder'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded successfully from {load_path}")
    return checkpoint['epoch'], checkpoint['mlm_loss'], checkpoint['sfbo_loss'], checkpoint['mpm_loss']

# Gradient accumulation steps
accumulation_steps = 5
accumulated_mlm_loss = 0.0  # Accumulate mlm_loss
accumulated_sfbo_loss = 0.0  # Accumulate sfbo_loss
batch_counter = 0  # To count batches for accumulation

def backward_and_optimize(accumulated_mlm_loss, accumulated_sfbo_loss):
    """Perform backpropagation and optimization on accumulated losses."""
    total_accumulated_loss = accumulated_mlm_loss + accumulated_sfbo_loss
    print(total_accumulated_loss)
    optimizer.zero_grad()
    total_accumulated_loss.backward()
    optimizer.step()

for epoch in range(10):
    print("Run", epoch)
    previous_packet_file = None
    all_packet_encodings = []
    previous_packet_number = None
    total_packet_enc_loss = 0

    for i, (packet_sequences, field_position, header_position, file_name) in enumerate(train_loader):
        # Move data to the device
        packet_sequences = packet_sequences.squeeze(0).to(device)
        field_position = field_position.squeeze(0).to(device)
        header_position = header_position.squeeze(0).to(device)

        current_file_name = file_name[0]

        # If starting a new file, process the previous file
        if previous_packet_file is not None and current_file_name != previous_packet_file:
            if all_packet_encodings:
                print(f"Completed processing file: {previous_packet_file}")
                batch_counter = 0
                # Process encodings and compute total_loss
                mpm_loss = process_encodings(all_packet_encodings, previous_packet_number)
                # total_length = len(all_packet_encodings)
                # avg_packet_enc_loss = total_packet_enc_loss / total_length
                # total_loss = mpm_loss

                # Perform standard optimization for the file
                optimizer.zero_grad()
                mpm_loss.backward()
                optimizer.step()

                # Clean up
                # del mpm_loss, mlm_loss, sfbo_loss, encoded_packets_mean
                torch.cuda.empty_cache()

            # Reset file-specific variables
            all_packet_encodings = []
            total_packet_enc_loss = 0
            previous_packet_number = None
            print(f"Started processing new file: {current_file_name}")

        previous_packet_file = current_file_name

        # Generate packet embeddings
        mlm_loss, sfbo_loss, encoded_packets_mean = packet_encoder(
            packet_sequences, field_pos=field_position, header_pos=header_position
        )
        torch.cuda.empty_cache()

        # Accumulate mlm_loss and sfbo_loss
        accumulated_mlm_loss += mlm_loss
        accumulated_sfbo_loss += sfbo_loss
        batch_counter += 1

        # Perform optimization after reaching accumulation_steps
        if batch_counter == accumulation_steps:
            backward_and_optimize(accumulated_mlm_loss, accumulated_sfbo_loss)
            accumulated_mlm_loss = 0.0
            accumulated_sfbo_loss = 0.0
            batch_counter = 0

        # total_packet_enc_loss += (mlm_loss + sfbo_loss)
        all_packet_encodings.append(encoded_packets_mean.detach())

        print(torch.cuda.memory_summary())
        print(f"Batch {i+1}")
        print(file_name)

        # Update packet number
        match = re.search(r'packet_(\d+)\.txt', current_file_name)
        if match:
            current_packet_number = match.group(1)

            if previous_packet_number is not None and current_packet_number != previous_packet_number:
                print(f"Flow completed for packet: {previous_packet_number}")
                all_packet_encodings = []

            previous_packet_number = current_packet_number
        else:
            print("No packet number found in file name:", current_file_name)

        torch.cuda.empty_cache()
    
    model_state = {
        'packet_embedding': packet_embedding.state_dict(),
        'packet_encoder': packet_encoder.state_dict(),
        'flow_embedding': flow_embedding.state_dict(),
        'flow_encoder': flow_encoder.state_dict()
    }
    save_model(f"final_model_{epoch+1}.pth", epoch+1, model_state, optimizer, accumulated_mlm_loss.item(), accumulated_sfbo_loss.item(), mpm_loss.item())
    
    # Final processing for the last file
    if all_packet_encodings and previous_packet_number is not None:
        print(f"Final processing for last flow packet: {previous_packet_number} in file: {previous_packet_file}")

        mpm_loss = process_encodings(all_packet_encodings, previous_packet_number)
        # total_length = len(all_packet_encodings)
        # avg_packet_enc_loss = total_packet_enc_loss / total_length
        # total_loss = mpm_loss + avg_packet_enc_loss

        # Perform standard optimization for the last file
        optimizer.zero_grad()
        mpm_loss.backward()
        optimizer.step()
    
    

    # Final optimization for remaining accumulated loss
    # if batch_counter > 0:
    #     backward_and_optimize()
