def test_compute_mfcc():
    import torchaudio
    import torch
    import numpy as np
    from torchaudio.transforms import MFCC
    import logging

    class TestMFCC:
        def __init__(self, sampling_rate=16000, n_mfcc=13):
            self.sampling_rate = sampling_rate
            self.mfcc_transform = MFCC(
                sample_rate=sampling_rate,
                n_mfcc=n_mfcc
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        def compute_full_mfcc(self, audio_path, segment_duration=10.0):
            """
            音声を小さなセグメントに分割してMFCCを計算し、メモリ使用量を削減します。
            """
            torch.cuda.empty_cache()
            try:
                waveform, file_sr = torchaudio.load(audio_path)

                # サンプリングレートの変換
                if file_sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=self.sampling_rate)
                    waveform = resampler(waveform)

                # デバイスに移動
                waveform = waveform.to(self.device)

                # セグメントごとに処理
                total_duration = waveform.shape[1] / self.sampling_rate
                segments = []
                max_length = 0

                for start in np.arange(0, total_duration, segment_duration):
                    end = min(start + segment_duration, total_duration)
                    start_sample = int(start * self.sampling_rate)
                    end_sample = int(end * self.sampling_rate)
                    segment_waveform = waveform[:, start_sample:end_sample]

                    if segment_waveform.shape[1] > 0:
                        mfcc = self.mfcc_transform(segment_waveform).squeeze(0)
                        segments.append(mfcc)
                        max_length = max(max_length, mfcc.size(1))
                        del segment_waveform, mfcc

                # セグメントのサイズを確認
                for idx, segment in enumerate(segments):
                    print(f"Segment {idx}: {segment.size()}")

                # セグメントをパディングして結合
                padded_segments = [
                    torch.nn.functional.pad(segment, (0, max_length - segment.size(1)))
                    for segment in segments
                ]

                full_mfcc = torch.cat(padded_segments, dim=1)
                torch.cuda.empty_cache()
                return full_mfcc

            except Exception as e:
                print(f"MFCC計算中にエラーが発生しました: {e}")
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
