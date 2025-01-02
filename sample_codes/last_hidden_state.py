from transformers import WavLMModel
import torch

def main():
    model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    input_values = torch.rand(1, 16000)  # Dummy data
    outputs = model(input_values)

    # Type and access
    print(outputs)
    print(type(outputs))  # transformers.modeling_outputs.BaseModelOutput
    print(outputs.last_hidden_state.shape)  # (batch_size, sequence_length, hidden_size)

if __name__ == "__main__":
    main()
