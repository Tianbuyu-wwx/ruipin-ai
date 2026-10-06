"""
测试声纹分析模块的日志输出
验证语速分析、停顿分析和情绪识别的详细日志是否正常打印

包含多种测试场景:
1. 短停顿场景 (<0.5s) - 正常呼吸/思考
2. 中停顿场景 (0.5-2s) - 较明显的思考或犹豫
3. 长停顿场景 (>2s) - 明显的卡壳/紧张
4. 混合停顿场景 - 多种停顿混合
5. 快速语速场景 - 语速偏快
6. 慢速语速场景 - 语速偏慢
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.voice_analysis.voice_analysis_manager import VoiceAnalysisManager


def generate_mock_speech(duration=5.0, sample_rate=16000, base_freq=150, syllable_rate=4):
    """
    生成模拟语音音频

    Args:
        duration: 语音时长(秒)
        sample_rate: 采样率
        base_freq: 基频(Hz)，影响音调
        syllable_rate: 每秒音节数，影响语速感
    """
    harmonics = [1, 2, 3, 4, 5]
    amplitudes = [0.4, 0.2, 0.1, 0.05, 0.02]

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.zeros_like(t)

    for h, amp in zip(harmonics, amplitudes):
        freq = base_freq * h
        freq_variation = 10 * np.sin(2 * np.pi * 3 * t)
        phase = 2 * np.pi * np.cumsum((freq + freq_variation) / sample_rate)
        audio += amp * np.sin(phase)

    envelope = np.ones_like(t)
    for i in range(int(duration * syllable_rate)):
        start = int(i * sample_rate / syllable_rate)
        end = int((i + 0.7) * sample_rate / syllable_rate)
        if end < len(envelope):
            envelope[start:end] = 1.0
            fade_len = min(100, end - start)
            envelope[start:start + fade_len] *= np.linspace(0, 1, fade_len)
            envelope[end - fade_len:end] *= np.linspace(1, 0, fade_len)
        if start < len(envelope):
            envelope[end:min(end + int(0.3 * sample_rate / syllable_rate), len(envelope))] = 0.1

    audio *= envelope
    noise = np.random.normal(0, 0.01, len(audio))
    audio += noise
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio


def create_test_scenario(name, segments, sample_rate=16000):
    """
    创建测试场景音频

    Args:
        name: 场景名称
        segments: 音频段列表，每个元素是 (类型, 时长) 或 (类型, 时长, 参数)
            类型: 'speech' 或 'silence'
            参数: speech时可传入syllable_rate等

    Returns:
        (音频数组, 对应文字)
    """
    audio_parts = []
    total_duration = 0
    word_count = 0

    for seg in segments:
        seg_type = seg[0]
        duration = seg[1]

        if seg_type == 'speech':
            kwargs = seg[2] if len(seg) > 2 else {}
            audio = generate_mock_speech(duration=duration, sample_rate=sample_rate, **kwargs)
            # 估算字数: 每秒约4个字
            word_count += int(duration * 4)
        else:  # silence
            audio = np.zeros(int(sample_rate * duration))

        audio_parts.append(audio)
        total_duration += duration

    full_audio = np.concatenate(audio_parts)

    # 生成对应文字
    text = f"这是{name}的测试音频"

    return full_audio, text, word_count


def run_scenario(manager, scenario_name, audio, text, expected_pauses):
    """
    运行单个测试场景

    Args:
        manager: VoiceAnalysisManager 实例
        scenario_name: 场景名称
        audio: 音频数据
        text: 转录文字
        expected_pauses: 期望的停顿信息
    """
    print("\n" + "=" * 70)
    print(f"【测试场景】{scenario_name}")
    print("=" * 70)
    print(f"音频时长: {len(audio)/16000:.2f}s | 文字: {text}")
    print(f"预期停顿: {expected_pauses}")
    print("-" * 70)

    report = manager.analyze(
        audio=audio,
        sr=16000,
        transcribed_text=text
    )

    print("-" * 70)
    print("【结果验证】")
    pause_breakdown = report['pause']['pause_breakdown']
    print(f"  实际停顿: 短={pause_breakdown['short']}, 中={pause_breakdown['medium']}, 长={pause_breakdown['long']}")
    print(f"  语速: {report['speech_rate']['wcpm']:.1f} WCPM ({report['speech_rate']['category']})")
    print(f"  流畅度: {report['pause']['score']:.1f}")
    print(f"  情绪: {report['emotion']['primary_emotion']} (置信度: {report['emotion']['confidence']:.1f}%)")

    return report


def test_all_scenarios():
    """测试所有场景"""
    print("=" * 70)
    print("声纹分析 - 多场景停顿类型验证测试")
    print("=" * 70)

    manager = VoiceAnalysisManager(language="zh")

    scenarios = []

    # 场景1: 短停顿场景 (<0.5s) - 正常呼吸
    # 语音段之间插入0.3秒静音
    audio, text, _ = create_test_scenario(
        "短停顿场景",
        [
            ('speech', 2.0),
            ('silence', 0.3),  # 短停顿 < 0.5s
            ('speech', 2.0),
            ('silence', 0.2),  # 短停顿 < 0.5s
            ('speech', 2.0),
        ]
    )
    scenarios.append(("场景1: 短停顿场景 (正常呼吸)", audio, text,
                      "预期: 2个短停顿, 0个中停顿, 0个长停顿"))

    # 场景2: 中停顿场景 (0.5-2s) - 思考犹豫
    audio, text, _ = create_test_scenario(
        "中停顿场景",
        [
            ('speech', 2.0),
            ('silence', 1.0),  # 中停顿 0.5-2s
            ('speech', 2.0),
            ('silence', 1.5),  # 中停顿 0.5-2s
            ('speech', 2.0),
        ]
    )
    scenarios.append(("场景2: 中停顿场景 (思考犹豫)", audio, text,
                      "预期: 0个短停顿, 2个中停顿, 0个长停顿"))

    # 场景3: 长停顿场景 (>2s) - 明显卡壳
    audio, text, _ = create_test_scenario(
        "长停顿场景",
        [
            ('speech', 2.0),
            ('silence', 2.5),  # 长停顿 > 2s
            ('speech', 2.0),
            ('silence', 3.0),  # 长停顿 > 2s
            ('speech', 2.0),
        ]
    )
    scenarios.append(("场景3: 长停顿场景 (明显卡壳/紧张)", audio, text,
                      "预期: 0个短停顿, 0个中停顿, 2个长停顿"))

    # 场景4: 混合停顿场景 - 短+中+长都有
    audio, text, _ = create_test_scenario(
        "混合停顿场景",
        [
            ('speech', 2.0),
            ('silence', 0.3),  # 短停顿
            ('speech', 1.5),
            ('silence', 1.0),  # 中停顿
            ('speech', 1.5),
            ('silence', 2.5),  # 长停顿
            ('speech', 2.0),
        ]
    )
    scenarios.append(("场景4: 混合停顿场景 (短+中+长)", audio, text,
                      "预期: 1个短停顿, 1个中停顿, 1个长停顿"))

    # 场景5: 快速语速场景 - 语速偏快
    audio, text, _ = create_test_scenario(
        "快速语速场景",
        [
            ('speech', 3.0, {'syllable_rate': 8}),  # 高速音节
            ('silence', 0.5),
            ('speech', 3.0, {'syllable_rate': 8}),
        ]
    )
    scenarios.append(("场景5: 快速语速场景", audio, text,
                      "预期: 语速偏快 (WCPM > 200)"))

    # 场景6: 慢速语速场景 - 语速偏慢
    audio, text, _ = create_test_scenario(
        "慢速语速场景",
        [
            ('speech', 5.0, {'syllable_rate': 2}),  # 低速音节
            ('silence', 1.0),
            ('speech', 5.0, {'syllable_rate': 2}),
        ]
    )
    scenarios.append(("场景6: 慢速语速场景", audio, text,
                      "预期: 语速偏慢 (WCPM < 120)"))

    # 运行所有场景
    results = []
    for name, audio, text, expected in scenarios:
        report = run_scenario(manager, name, audio, text, expected)
        results.append((name, report))

    # 汇总
    print("\n" + "=" * 70)
    print("【测试汇总】")
    print("=" * 70)
    for name, report in results:
        pause = report['pause']
        emotion = report['emotion']
        print(f"\n{name}:")
        print(f"  语速: {report['speech_rate']['wcpm']:.1f} WCPM ({report['speech_rate']['category']})")
        print(f"  停顿: 短={pause['pause_breakdown']['short']}, 中={pause['pause_breakdown']['medium']}, 长={pause['pause_breakdown']['long']}")
        print(f"  流畅度: {pause['score']:.1f}")
        print(f"  情绪: {emotion['primary_emotion']} (置信度: {emotion['confidence']:.1f}%, 强度: {emotion['intensity']:.1f})")

    print("\n" + "=" * 70)
    print("所有场景测试完成!")
    print("请检查上方日志输出，验证:")
    print("  1. 短停顿(<0.5s)是否被正确分类")
    print("  2. 中停顿(0.5-2s)是否被正确分类")
    print("  3. 长停顿(>2s)是否被正确分类")
    print("  4. 语速计算是否符合预期")
    print("  5. 流畅度评分是否合理")
    print("=" * 70)

    return results


if __name__ == "__main__":
    test_all_scenarios()
