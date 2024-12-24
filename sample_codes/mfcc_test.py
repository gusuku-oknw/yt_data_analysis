def test_compute_mfcc():
    import torchaudio
    import torch
    import numpy as np
    from torchaudio.transforms import MFCC

    class TestMFCC:
        def __init__(self, sampling_rate=16000, n_mfcc=13):
            self.sampling_rate = sampling_rate
            self.mfcc_transform = MFCC(
                sample_rate=sampling_rate,
                n_mfcc=n_mfcc
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        def compute_full_mfcc(self, audio_path, segment_duration=10.0):
            """
            Compute MFCC for audio in smaller segments to reduce memory usage.
            Handles padding to ensure all segments are of the same size.
            """
            torch.cuda.empty_cache()
            try:
                # Load the audio file
                waveform, file_sr = torchaudio.load(audio_path)

                # Resample if the sampling rate doesn't match
                if file_sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=self.sampling_rate)
                    waveform = resampler(waveform)

                # Move the waveform to the correct device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                waveform = waveform.to(device)

                # Calculate total duration
                total_duration = waveform.shape[1] / self.sampling_rate

                # Initialize variables
                segments = []

                # Process the waveform in smaller segments
                for start in np.arange(0, total_duration, segment_duration):
                    end = min(start + segment_duration, total_duration)
                    start_sample = int(start * self.sampling_rate)
                    end_sample = int(end * self.sampling_rate)
                    segment_waveform = waveform[:, start_sample:end_sample]

                    if segment_waveform.shape[1] > 0:  # Ensure the segment is not empty
                        mfcc = self.mfcc_transform(segment_waveform).squeeze(0)  # Compute MFCC
                        segments.append(mfcc)
                        del segment_waveform, mfcc  # Free memory

                if not segments:
                    raise RuntimeError("No valid segments were processed for MFCC computation.")

                # Debug: Log all segment sizes before padding
                print("Segment sizes before padding:")
                for idx, segment in enumerate(segments):
                    print(f"  Segment {idx}: {segment.size()}")

                # Determine the maximum length
                max_length = max(segment.size(1) for segment in segments)

                # Pad all segments to the maximum length
                padded_segments = []
                for idx, segment in enumerate(segments):
                    padded_segment = torch.nn.functional.pad(segment, (0, max_length - segment.size(1)))
                    padded_segments.append(padded_segment)

                # Verify segment sizes after padding
                print("Segment sizes after padding:")
                for idx, segment in enumerate(padded_segments):
                    print(f"  Segment {idx}: {segment.size()}")
                    if segment.size(1) != max_length:
                        print(f"Segment {idx} has incorrect size after padding: {segment.size(1)}")
                        raise RuntimeError(f"Padding failed for segment {idx}.")

                # Concatenate all padded segments along the time axis
                full_mfcc = torch.cat(padded_segments, dim=2)

                # Clear CUDA memory cache
                torch.cuda.empty_cache()

                return full_mfcc

            except torch.cuda.OutOfMemoryError as e:
                print("CUDA memory error during MFCC computation:", e)
                torch.cuda.empty_cache()
                raise
            except Exception as e:
                print("Error occurred during MFCC computation:", e)
                raise

    # Test setup
    test_audio_path = "../data/audio/source/bh4ObBry9q4.mp3"  # Replace with the path to a test audio file
    mfcc_tester = TestMFCC()

    try:
        full_mfcc = mfcc_tester.compute_full_mfcc(test_audio_path)
        print("MFCC computation successful. Shape:", full_mfcc.shape)
    except Exception as e:
        print("MFCC computation failed:", e)

if __name__ == "__main__":
    test_compute_mfcc()
