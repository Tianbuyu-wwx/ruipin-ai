"""
生成 mock 测试音频文件
用于测试 Whisper 音频转录功能的日志输出

支持生成:
1. 纯语音测试音频 (使用 numpy 生成模拟语音波形)
2. 带静音的测试音频
3. 多段语音的测试音频
"""

import numpy as np
import wave
import struct
from pathlib import Path


def generate_sine_wave(frequency, duration, sample_rate=16000, amplitude=0.5):
    """生成正弦波模拟语音"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave_data = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave_data


def generate_mock_speech(duration=5.0, sample_rate=16000):
    """
    生成模拟语音音频
    使用多个不同频率的正弦波叠加，模拟人声的频谱特征
    """
    # 模拟语音的频率范围 (85-255 Hz 基频，加上谐波)
    base_freq = 150  # 基频
    harmonics = [1, 2, 3, 4, 5]  # 谐波次数
    amplitudes = [0.4, 0.2, 0.1, 0.05, 0.02]  # 谐波幅度

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.zeros_like(t)

    # 添加基频和谐波
    for h, amp in zip(harmonics, amplitudes):
        freq = base_freq * h
        # 添加一些频率变化模拟语调
        freq_variation = 10 * np.sin(2 * np.pi * 3 * t)  # 3Hz 语调变化
        phase = 2 * np.pi * np.cumsum((freq + freq_variation) / sample_rate)
        audio += amp * np.sin(phase)

    # 添加包络模拟语音的浊音/清音特征
    envelope = np.ones_like(t)
    syllable_rate = 4  # 每秒音节数
    for i in range(int(duration * syllable_rate)):
        start = int(i * sample_rate / syllable_rate)
        end = int((i + 0.7) * sample_rate / syllable_rate)
        if end < len(envelope):
            envelope[start:end] = 1.0
            # 淡入淡出
            fade_len = min(100, end - start)
            envelope[start:start + fade_len] *= np.linspace(0, 1, fade_len)
            envelope[end - fade_len:end] *= np.linspace(1, 0, fade_len)
        if start < len(envelope):
            envelope[end:min(end + int(0.3 * sample_rate / syllable_rate), len(envelope))] = 0.1

    audio *= envelope

    # 添加轻微噪声模拟录音环境
    noise = np.random.normal(0, 0.01, len(audio))
    audio += noise

    # 归一化
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio


def add_silence(audio, sample_rate, silence_duration, position='end'):
    """在音频指定位置添加静音"""
    silence = np.zeros(int(sample_rate * silence_duration))
    if position == 'start':
        return np.concatenate([silence, audio])
    elif position == 'middle':
        mid = len(audio) // 2
        return np.concatenate([audio[:mid], silence, audio[mid:]])
    else:  # end
        return np.concatenate([audio, silence])


def save_wav(audio_data, filepath, sample_rate=16000):
    """保存为 WAV 文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 转换为 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)

    with wave.open(str(filepath), 'w') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"[OK] 音频已保存: {filepath}")
    print(f"     时长: {len(audio_data) / sample_rate:.2f} 秒")
    print(f"     采样率: {sample_rate} Hz")
    print(f"     文件大小: {filepath.stat().st_size / 1024:.2f} KB")


def generate_test_audio_suite(output_dir="tests/mock_audio"):
    """生成一套测试音频文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("生成 Mock 测试音频文件")
    print("=" * 60)

    # 1. 标准测试音频 (5秒模拟语音)
    print("\n[1/5] 生成标准测试音频 (5秒模拟语音)...")
    audio = generate_mock_speech(duration=5.0)
    save_wav(audio, output_dir / "test_speech_5s.wav")

    # 2. 短音频 (2秒)
    print("\n[2/5] 生成短测试音频 (2秒)...")
    audio = generate_mock_speech(duration=2.0)
    save_wav(audio, output_dir / "test_speech_2s.wav")

    # 3. 长音频 (10秒)
    print("\n[3/5] 生成长测试音频 (10秒)...")
    audio = generate_mock_speech(duration=10.0)
    save_wav(audio, output_dir / "test_speech_10s.wav")

    # 4. 带静音的音频 (中间有2秒静音)
    print("\n[4/5] 生成带静音的测试音频 (语音+静音)...")
    audio = generate_mock_speech(duration=3.0)
    audio = add_silence(audio, 16000, 2.0, position='middle')
    audio = add_silence(audio, 16000, 1.0, position='end')
    save_wav(audio, output_dir / "test_speech_with_silence.wav")

    # 5. 纯静音音频 (用于测试空内容处理)
    print("\n[5/5] 生成纯静音测试音频...")
    silence = np.zeros(16000 * 3)  # 3秒静音
    save_wav(silence, output_dir / "test_silence_3s.wav")

    print("\n" + "=" * 60)
    print("所有测试音频生成完成!")
    print(f"输出目录: {output_dir.absolute()}")
    print("=" * 60)

    return output_dir


def test_with_whisper(audio_path):
    """使用 Whisper 测试转录并打印日志"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.audio_processor import AudioProcessorManager

    print("\n" + "=" * 60)
    print(f"测试 Whisper 转录: {audio_path}")
    print("=" * 60)

    manager = AudioProcessorManager()

    result = manager.process_audio_input(
        audio_path=audio_path,
        text_input=None,
        language="zh"
    )

    print("\n" + "=" * 60)
    print("转录结果")
    print("=" * 60)
    print(f"转录成功: {result['has_transcription']}")
    print(f"转录文字: {result['transcribed_text'][:200]}...")
    print(f"合并文字: {result['combined_text'][:200]}...")
    print(f"转录详情: {result['transcription_info']}")
    if result.get('errors'):
        print(f"错误信息: {result['errors']}")


if __name__ == "__main__":
    # 生成测试音频
    audio_dir = generate_test_audio_suite()

    # 测试转录
    test_files = [
        audio_dir / "test_speech_5s.wav",
        audio_dir / "test_speech_2s.wav",
        audio_dir / "test_speech_with_silence.wav",
        audio_dir / "test_silence_3s.wav",
    ]

    for test_file in test_files:
        if test_file.exists():
            test_with_whisper(test_file)
