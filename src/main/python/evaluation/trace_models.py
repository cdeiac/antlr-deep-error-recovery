import torch

from models.lstm import LSTMDenoiser

if __name__ == '__main__':
    # hyperparams
    vocab_size = 130
    embedding_dim = 128
    hidden_size = 128
    num_layers = 1
    bidirectional = True
    # config
    model = LSTMDenoiser(vocab_size, embedding_dim, hidden_size, num_layers, bidirectional)

    base_path = "src/main/python/data/generated/checkpoints/"
    cv_dirs = ["00_001", "00_005", "00_010"]
    checkpoint_paths = ["/checkpoint0.pt", "/checkpoint1.pt", "/checkpoint2.pt"]
    for cv_dir in cv_dirs:
        for path in checkpoint_paths:
            checkpoint = torch.load(base_path + cv_dir + path)
            model.load_state_dict(checkpoint)
            model.eval()
            trace = torch.randint(0, 130, (512,))
            traced_script_module = torch.jit.trace(model, trace)
            traced_script_module.save(base_path + cv_dir + "/traced_model" + path.split('.')[0][-1] + ".pt")