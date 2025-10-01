# server.py - Whisper + KoNLPy + librosa 통합 음성 분석 시스템
# 설치: pip install mediapipe opencv-python flask flask-cors numpy pydub openai-whisper konlpy librosa soundfile

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import json
import time
from collections import deque, Counter
import threading
import random
import atexit
import io
import os

# AI 라이브러리
try:
    from pydub import AudioSegment
    AUDIO_PROCESSING_AVAILABLE = True
    print("✅ pydub 라이브러리 로드 완료")
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    print("⚠️ pydub 라이브러리가 없습니다.")

try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper 로드 완료")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper가 없습니다. pip install openai-whisper")

try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = False
    print("✅ KoNLPy 로드 완료")
except ImportError:
    KONLPY_AVAILABLE = False
    print("⚠️ KoNLPy가 없습니다. pip install konlpy")

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
    print("✅ librosa 로드 완료")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa가 없습니다. pip install librosa soundfile")

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose

# MediaPipe 초기화
try:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✅ MediaPipe Pose 초기화 완료")
except Exception as e:
    print(f"❌ MediaPipe Pose 초기화 실패: {e}")
    pose = None

# AI 모델 초기화
whisper_model = None
okt = None

if WHISPER_AVAILABLE:
    print("🔄 Whisper 모델 로딩 중... (최초 1회, 약 30초 소요)")
    whisper_model = whisper.load_model("tiny")  # tiny/base/small 중 선택
    print("✅ Whisper 모델 로드 완료")

if KONLPY_AVAILABLE:
    okt = Okt()
    print("✅ KoNLPy Okt 초기화 완료")

# 음성 분석 결과 저장
voice_analysis_data = {
    "session_active": False,
    "start_time": None,
    "audio_samples": [],
    "filler_words": [],
    "voice_confidence": [],
    "speaking_pace": [],
    "volume_levels": [],
    "total_speaking_time": 0,
    "silence_periods": [],
    "analysis_complete": False,
    "final_report": {},
    "chunk_count": 0,
    "ai_analysis": {}  # AI 분석 결과
}

# 추임새/필러 단어 목록 (확장)
FILLER_WORDS = [
    "어", "음", "아", "그", "뭐", "이제", "그래서", "그니까", "그런데",
    "어떻게", "뭔가", "약간", "좀", "그거", "이거", "저기", "아무튼",
    "일단", "우선", "그러면", "그럼", "아니", "맞아", "그치", "응", "네",
    "에", "엄", "흠", "아무래도"
]

# 접속사 목록
CONNECTIVES = [
    "그래서", "그러나", "하지만", "그리고", "또한", "따라서", "그런데",
    "왜냐하면", "즉", "결국", "그러므로", "그렇지만", "또는", "혹은",
    "그럼에도", "반면에", "한편", "더불어"
]

# 체육대학 육상 전공 면접 질문 데이터베이스
INTERVIEW_QUESTIONS = {
    "기본질문": [
        "간단하게 자기소개를 해주세요.",
        "체육학과에 지원한 동기는 무엇인가요?",
        "본인의 장점과 단점을 이야기해 주세요.",
        "가장 좋아하는 운동 종목은 무엇이고, 그 이유는 무엇인가요?",
        "대학교 4년 동안 무엇을 배우고 싶나요?"
    ],
    "지원동기": [
        "우리 학교 체육학과를 선택한 특별한 이유가 있나요?",
        "체육교사나 스포츠지도자가 되고 싶은 이유는 무엇인가요?",
        "졸업 후 진로 계획에 대해 구체적으로 말씀해주세요.",
        "체육 분야에서 본인만의 목표나 비전이 있다면 무엇인가요?",
        "이 분야를 선택하게 된 계기나 영향을 받은 사람이 있나요?"
    ],
    "육상전문": [
        "육상 운동을 시작하게 된 계기는 무엇인가요?",
        "본인이 가장 자신 있는 육상 종목은 무엇이고, 어떤 기록을 가지고 있나요?",
        "육상 운동의 매력은 무엇이라고 생각하나요?",
        "단거리와 장거리 육상의 차이점에 대해 설명해보세요.",
        "육상 선수에게 필요한 체력 요소는 무엇인가요?",
        "좋아하는 육상 선수가 있다면 누구이고 그 이유는 무엇인가요?"
    ],
    "상황질문": [
        "팀 내에서 갈등이 생겼을 때 어떻게 해결하시겠나요?",
        "중요한 대회를 앞두고 부상을 당했다면 어떻게 하시겠나요?",
        "훈련이 힘들어서 포기하고 싶을 때는 어떻게 극복하나요?",
        "목표를 달성하지 못했을 때의 경험과 극복 방법을 말해보세요.",
        "후배나 동료와 의견 차이가 있을 때 어떻게 해결하나요?",
        "실수를 했을 때 어떻게 대처하나요?"
    ],
    "압박질문": [
        "다른 지원자들과 비교했을 때 본인의 경쟁력은 무엇인가요?",
        "체육 성적이 좋지 않다면 그 이유는 무엇인가요?",
        "운동을 싫어하는 학생에게 어떻게 동기부여를 할 것인가요?",
        "체육 수업 시간에 다친 학생이 있다면 어떻게 대처하시겠나요?",
        "본인의 가장 큰 약점은 무엇이고, 어떻게 개선하고 있나요?",
        "스트레스를 받을 때 어떻게 해소하시나요?"
    ],
    "마무리질문": [
        "우리 학교에 입학하면 가장 먼저 하고 싶은 일은 무엇인가요?",
        "10년 후 본인의 모습을 어떻게 그려보고 있나요?",
        "체육 지도자로서 가장 보람 있을 것 같은 순간은 언제인가요?",
        "마지막으로 하고 싶은 말씀이 있다면 해주세요.",
        "우리에게 꼭 물어보고 싶은 것이 있나요?"
    ]
}

INTERVIEW_FLOW = [
    "기본질문", "지원동기", "육상전문", "상황질문", "압박질문", "마무리질문"
]

# 면접 세션 관리
interview_session = {
    "active": False,
    "current_question": None,
    "question_index": 0,
    "category": None,
    "questions_list": [],
    "start_time": None,
    "thinking_time": 10,
    "answer_time": 5,
    "phase": "thinking",
    "auto_timer": None,
    "phase_start_time": None,
    "current_flow_stage": 0
}

# ==================== AI 음성 분석 함수들 ====================

def analyze_voice_tremor_librosa(audio_path):
    """librosa로 음정 떨림 분석"""
    if not LIBROSA_AVAILABLE:
        print("⚠️ librosa가 설치되지 않아 기본 분석 사용")
        return None
    
    try:
        print(f"🔬 librosa 음정 떨림 분석 시작: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None)
        
        if len(y) < sr * 0.5:  # 0.5초 미만이면 스킵
            print("⚠️ 오디오가 너무 짧음")
            return None
        
        # 피치 추출
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=300)
        pitch_values = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 1:
            pitch_changes = np.diff(pitch_values)
            jitter = np.std(pitch_changes)
            jitter_percent = (jitter / np.mean(pitch_values)) * 100
        else:
            jitter = 0
            jitter_percent = 0
        
        # 음량 떨림 (Shimmer)
        rms = librosa.feature.rms(y=y)[0]
        shimmer = np.std(rms)
        
        # 스펙트럴 안정성
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_stability = np.std(spectral_centroid)
        
        # 자신감 점수 계산
        if jitter_percent < 1.0 and shimmer < 0.02:
            confidence = 95
        elif jitter_percent < 2.0 and shimmer < 0.04:
            confidence = 80
        elif jitter_percent < 3.5 and shimmer < 0.06:
            confidence = 60
        else:
            confidence = 40
        
        result = {
            'jitter': float(jitter),
            'jitter_percent': float(jitter_percent),
            'shimmer': float(shimmer),
            'spectral_stability': float(spectral_stability),
            'average_pitch': float(np.mean(pitch_values)) if pitch_values else 0,
            'pitch_range': float(np.max(pitch_values) - np.min(pitch_values)) if len(pitch_values) > 1 else 0,
            'voice_confidence': confidence
        }
        
        print(f"✅ librosa 분석 완료: Jitter={jitter_percent:.2f}%, 자신감={confidence}")
        return result
        
    except Exception as e:
        print(f"❌ librosa 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_speech_with_whisper_konlpy(audio_path):
    """Whisper + KoNLPy 종합 텍스트 분석"""
    if not WHISPER_AVAILABLE:
        print("⚠️ Whisper가 설치되지 않아 텍스트 분석 불가")
        return None
    
    try:
        print(f"🎤 Whisper 음성 인식 시작: {audio_path}")
        
        # Whisper로 음성 인식
        result = whisper_model.transcribe(audio_path, language='ko')
        full_text = result['text']
        
        print(f"📝 인식된 텍스트 ({len(full_text)}자): {full_text[:100]}...")
        
        if not full_text.strip():
            print("⚠️ 인식된 텍스트가 없음")
            return None
        
        # 기본 단어 분석
        words = full_text.split()
        
        # 추임새 분석
        filler_count = Counter()
        for word in words:
            clean_word = word.strip('.,!?').lower()
            if clean_word in FILLER_WORDS:
                filler_count[clean_word] += 1
        
        # 접속사 분석
        connective_count = Counter()
        for word in words:
            clean_word = word.strip('.,!?').lower()
            if clean_word in CONNECTIVES:
                connective_count[clean_word] += 1
        
        # KoNLPy 형태소 분석
        nouns = []
        verbs = []
        adjectives = []
        adverbs = []
        
        if KONLPY_AVAILABLE and okt:
            print("🔍 KoNLPy 품사 분석 시작...")
            try:
                morphs = okt.pos(full_text)
                nouns = [word for word, pos in morphs if pos == 'Noun']
                verbs = [word for word, pos in morphs if pos == 'Verb']
                adjectives = [word for word, pos in morphs if pos == 'Adjective']
                adverbs = [word for word, pos in morphs if pos == 'Adverb']
                print(f"✅ KoNLPy 분석 완료: 명사={len(nouns)}, 동사={len(verbs)}")
            except Exception as e:
                print(f"⚠️ KoNLPy 분석 실패: {e}")
        
        # 반복 단어 분석
        word_freq = Counter(word.strip('.,!?').lower() for word in words if len(word) > 1)
        repeated_words = {word: count for word, count in word_freq.items() if count > 2}
        
        # 문장 분석
        sentences = [s.strip() for s in full_text.split('.') if s.strip()]
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # 어휘 다양성
        vocabulary_richness = len(set(nouns)) / len(nouns) if nouns else 0
        
        analysis_result = {
            'full_text': full_text,
            'total_words': len(words),
            
            # 추임새
            'filler_words': dict(filler_count),
            'filler_total': sum(filler_count.values()),
            'filler_ratio': (sum(filler_count.values()) / len(words) * 100) if words else 0,
            
            # 접속사
            'connectives': dict(connective_count),
            'connective_total': sum(connective_count.values()),
            
            # 품사별
            'noun_count': len(nouns),
            'verb_count': len(verbs),
            'adjective_count': len(adjectives),
            'adverb_count': len(adverbs),
            'top_nouns': dict(Counter(nouns).most_common(10)),
            
            # 반복 단어
            'repeated_words': repeated_words,
            
            # 문장
            'sentence_count': len(sentences),
            'avg_sentence_length': avg_sentence_length,
            
            # 어휘력
            'vocabulary_richness': vocabulary_richness * 100,
            
            # 시간별 세그먼트
            'segments': result.get('segments', [])
        }
        
        print(f"✅ Whisper+KoNLPy 분석 완료")
        print(f"   - 추임새: {analysis_result['filler_total']}회 ({analysis_result['filler_ratio']:.1f}%)")
        print(f"   - 접속사: {analysis_result['connective_total']}회")
        print(f"   - 어휘 다양성: {analysis_result['vocabulary_richness']:.1f}%")
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ Whisper+KoNLPy 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_ai_recommendations(tremor, speech):
    """AI 분석 결과 기반 추천사항 생성"""
    recommendations = []
    
    if tremor:
        # 음성 떨림
        if tremor['jitter_percent'] > 3.0:
            recommendations.append(f"목소리에 떨림이 있습니다 (Jitter: {tremor['jitter_percent']:.1f}%). 심호흡을 하고 천천히 말해보세요.")
        elif tremor['jitter_percent'] > 2.0:
            recommendations.append("약간의 긴장이 느껴집니다. 편안하게 답변해보세요.")
        
        if tremor['shimmer'] > 0.05:
            recommendations.append("음량 변화가 큽니다. 일정한 크기로 말해보세요.")
    
    if speech:
        # 추임새
        if speech['filler_ratio'] > 10:
            recommendations.append(f"추임새가 많습니다 ({speech['filler_total']}회, {speech['filler_ratio']:.1f}%). 의식적으로 줄여보세요.")
            if speech['filler_words']:
                top_fillers = sorted(speech['filler_words'].items(), key=lambda x: x[1], reverse=True)[:3]
                recommendations.append(f"가장 많이 사용한 추임새: {', '.join([f'{w}({c}회)' for w, c in top_fillers])}")
        elif speech['filler_ratio'] > 5:
            recommendations.append("추임새를 조금 줄이면 더 좋습니다.")
        
        # 접속사
        if speech['connective_total'] > 15:
            recommendations.append(f"접속사 사용이 많습니다 ({speech['connective_total']}회). 문장을 간결하게 만들어보세요.")
        
        # 반복 단어
        if speech.get('repeated_words'):
            top_repeated = sorted(speech['repeated_words'].items(), key=lambda x: x[1], reverse=True)[:3]
            if top_repeated:
                recommendations.append(f"반복된 단어: {', '.join([f'{w}({c}회)' for w, c in top_repeated])}")
        
        # 어휘 다양성
        if speech['vocabulary_richness'] < 30:
            recommendations.append("다양한 어휘를 사용해보세요.")
        elif speech['vocabulary_richness'] > 70:
            recommendations.append("어휘 사용이 풍부합니다!")
        
        # 문장 길이
        if speech['avg_sentence_length'] > 20:
            recommendations.append(f"문장이 깁니다 (평균 {speech['avg_sentence_length']:.1f}단어). 짧고 명확하게 말해보세요.")
        elif speech['avg_sentence_length'] < 5:
            recommendations.append("문장이 너무 짧습니다. 조금 더 자세히 설명해보세요.")
    
    if not recommendations:
        recommendations.append("훌륭한 답변입니다!")
        if tremor:
            recommendations.append("목소리가 안정적이고 명확합니다.")
        if speech:
            recommendations.append("언어 사용이 적절합니다.")
    
    return recommendations

# ==================== 기존 음성 분석 함수들 ====================

def analyze_audio_chunk(audio_data, sample_rate=16000):
    """실시간 오디오 청크 분석"""
    try:
        volume = np.sqrt(np.mean(audio_data**2))
        is_speaking = volume > 0.01
        
        if len(audio_data) > 512:
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
            magnitude = np.abs(fft)
            fundamental_freq = freqs[np.argmax(magnitude[1:len(magnitude)//2]) + 1] if len(magnitude) > 1 else 0
            voice_tremor = np.std(magnitude[1:100]) if len(magnitude) > 100 else 0
        else:
            fundamental_freq = 0
            voice_tremor = 0
        
        return {
            'volume': float(volume),
            'is_speaking': is_speaking,
            'fundamental_freq': float(fundamental_freq),
            'voice_tremor': float(voice_tremor),
            'timestamp': time.time()
        }
    except Exception as e:
        print(f"음성 분석 오류: {e}")
        return None

def generate_voice_analysis_report():
    """최종 음성 분석 리포트 생성 (AI 분석 포함)"""
    try:
        sample_count = len(voice_analysis_data["audio_samples"])
        
        print(f"📊 리포트 생성 시작: {sample_count}개 샘플")
        
        # 기본 분석
        if sample_count < 10:
            print(f"⚠️ 샘플 부족: {sample_count}개")
        
        volumes = [sample['volume'] for sample in voice_analysis_data["audio_samples"] if sample]
        speaking_periods = [sample for sample in voice_analysis_data["audio_samples"] if sample and sample.get('is_speaking', False)]
        voice_tremors = [sample.get('voice_tremor', 0) for sample in speaking_periods if sample.get('voice_tremor', 0) > 0]
        
        total_time = time.time() - voice_analysis_data["start_time"] if voice_analysis_data["start_time"] else 0
        speaking_time = len(speaking_periods) * 0.1
        speaking_ratio = speaking_time / total_time if total_time > 0 else 0
        
        avg_volume = np.mean(volumes) if volumes else 0
        volume_std = np.std(volumes) if len(volumes) > 1 else 0
        volume_consistency = max(0, 1 - (volume_std / (avg_volume + 0.001)))
        
        avg_tremor = np.mean(voice_tremors) if voice_tremors else 0
        tremor_score = max(0, 1 - (avg_tremor / 0.1))
        
        confidence_score = (
            (min(avg_volume * 100, 100) * 0.30) +
            (volume_consistency * 100 * 0.25) +
            (tremor_score * 100 * 0.25) +
            (speaking_ratio * 100 * 0.20)
        )
        
        confidence_score = min(100, max(0, confidence_score))
        
        # AI 분석 점수 반영
        ai_analysis = voice_analysis_data.get("ai_analysis", {})
        
        if ai_analysis:
            print("🤖 AI 분석 결과 통합 중...")
            
            # librosa 음성 안정성 점수
            if ai_analysis.get('tremor'):
                tremor_confidence = ai_analysis['tremor']['voice_confidence']
                confidence_score = (confidence_score * 0.5) + (tremor_confidence * 0.5)
                print(f"   - librosa 음성 안정성: {tremor_confidence}")
            
            # Whisper 추임새 점수
            if ai_analysis.get('speech'):
                filler_penalty = min(ai_analysis['speech']['filler_ratio'], 20)
                confidence_score = confidence_score * (1 - filler_penalty / 100)
                print(f"   - 추임새 비율: {ai_analysis['speech']['filler_ratio']:.1f}%")
        
        # 추천사항 생성
        recommendations = []
        
        if avg_volume < 0.3:
            recommendations.append("목소리가 작습니다. 더 자신있게 말해보세요.")
        elif avg_volume > 0.8:
            recommendations.append("목소리가 너무 큽니다. 조금 낮춰보세요.")
        
        if volume_consistency < 0.7:
            recommendations.append("목소리 크기가 일정하지 않습니다.")
        
        if avg_tremor > 0.05:
            recommendations.append("음성에 떨림이 감지됩니다. 심호흡 후 천천히 말해보세요.")
        
        if speaking_ratio < 0.6:
            recommendations.append("말하는 시간이 부족합니다. 더 적극적으로 답변해보세요.")
        
        # AI 추천사항 추가
        if ai_analysis:
            ai_recommendations = generate_ai_recommendations(
                ai_analysis.get('tremor'),
                ai_analysis.get('speech')
            )
            recommendations.extend(ai_recommendations)
        
        if not recommendations:
            recommendations.append("전반적으로 안정적인 음성으로 답변하셨습니다!")
        
# 약 550번째 줄 부근
        report = {
            "overall_score": round(confidence_score, 1),
            "detailed_analysis": {
                "voice_confidence": round(confidence_score, 1),
                "average_volume": round(avg_volume * 100, 1),
                "volume_consistency": round(volume_consistency * 100, 1),
                "voice_stability": round(tremor_score * 100, 1),
                "speaking_ratio": round(speaking_ratio * 100, 1),
                "total_speaking_time": round(speaking_time, 1),
                "filler_word_count": 0,  # 기본값
                "filler_ratio": 0  # 기본값
            },
            "recommendations": recommendations[:10],
            "analysis_timestamp": time.time(),
            "ai_powered": bool(ai_analysis),
            "debug_info": {
                "samples_analyzed": sample_count,
                "chunks_received": voice_analysis_data.get("chunk_count", 0),
                "speaking_periods": len(speaking_periods),
                "has_librosa": ai_analysis.get('tremor') is not None,
                "has_whisper": ai_analysis.get('speech') is not None
            }
        }

        # 🔧 AI 상세 분석 추가 (여기가 중요!)
        if ai_analysis.get('speech'):
            speech = ai_analysis['speech']
            
            # detailed_analysis에 AI 데이터 추가
            report["detailed_analysis"]["filler_word_count"] = speech['filler_total']
            report["detailed_analysis"]["filler_ratio"] = round(speech['filler_ratio'], 1)
            report["detailed_analysis"]["connective_count"] = speech['connective_total']
            report["detailed_analysis"]["vocabulary_richness"] = round(speech['vocabulary_richness'], 1)
            report["detailed_analysis"]["avg_sentence_length"] = round(speech['avg_sentence_length'], 1)
            
            # ai_details에 상세 정보 추가
            report["ai_details"] = {
                "recognized_text": speech['full_text'][:500] + "..." if len(speech['full_text']) > 500 else speech['full_text'],
                "filler_words": speech['filler_words'],
                "connectives": speech['connectives'],
                "top_nouns": speech['top_nouns']
            }

        if ai_analysis.get('tremor'):
            tremor = ai_analysis['tremor']
            report["detailed_analysis"]["jitter_percent"] = round(tremor['jitter_percent'], 2)
            report["detailed_analysis"]["shimmer"] = round(tremor['shimmer'], 4)
            report["detailed_analysis"]["average_pitch"] = round(tremor['average_pitch'], 1)

        voice_analysis_data["final_report"] = report
        voice_analysis_data["analysis_complete"] = True

        print(f"✅ 음성 분석 리포트 생성 완료: {confidence_score:.1f}점")
        if ai_analysis:
            print(f"   🤖 AI 분석 적용됨")
            if ai_analysis.get('speech'):
                print(f"      - 추임새: {speech['filler_total']}회 ({speech['filler_ratio']:.1f}%)")
                print(f"      - 인식된 텍스트: {len(speech['full_text'])}자")

        return report
        
    except Exception as e:
        print(f"❌ 음성 분석 리포트 생성 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "overall_score": 0,
            "detailed_analysis": {
                "voice_confidence": 0,
                "volume_consistency": 0,
                "voice_confidence": 0,
                "volume_consistency": 0,
                "voice_stability": 0,
                "speaking_ratio": 0,
                "total_speaking_time": 0,
                "filler_word_count": 0,
                "filler_ratio": 0
            },
            "recommendations": [
                f"분석 중 오류가 발생했습니다: {str(e)}",
                "서버 로그를 확인해주세요."
            ],
            "error": str(e)
        }

# ==================== 카메라 초기화 ====================

def initialize_camera():
    """내장 카메라부터 순차적으로 탐색 후 초기화"""
    print("📱 카메라 초기화 중...")
    camera = None

    for camera_index in range(5):
        print(f"🔍 카메라 {camera_index}번 시도...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ 카메라 {camera_index}번 연결 성공")
                camera = cap
                break
        cap.release()

    if not camera:
        print("❌ 사용 가능한 카메라가 없습니다.")
        return None

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)

    print("🎥 카메라 준비 완료")
    return camera

camera = initialize_camera()
if camera is None:
    print("💥 카메라를 초기화할 수 없습니다.")
    exit(1)

def play_beep():
    try:
        import os
        if os.name == 'posix':
            os.system('afplay /System/Library/Sounds/Ping.aiff &')
        elif os.name == 'nt':
            import winsound
            winsound.Beep(1000, 200)
    except:
        print('\a')

# ==================== 면접 질문 생성 ====================

def generate_structured_interview_questions():
    """흐름에 맞는 체계적 면접 질문 생성"""
    structured_questions = []
    
    first_question = {
        'category': '기본질문',
        'question': '간단하게 자기소개를 해주세요.',
        'stage': 0
    }
    structured_questions.append(first_question)
    
    for stage_idx, category in enumerate(INTERVIEW_FLOW):
        if category == "기본질문":
            continue
        
        available_questions = INTERVIEW_QUESTIONS[category]
        selected_question = random.choice(available_questions)
        
        structured_questions.append({
            'category': category,
            'question': selected_question,
            'stage': stage_idx
        })
    
    print("📋 체계적 면접 질문 구성:")
    for i, q in enumerate(structured_questions, 1):
        print(f"   {i}. [{q['category']}] {q['question'][:40]}...")
    
    return structured_questions

# ==================== 면접 자동 진행 ====================

def auto_next_phase():
    global interview_session
    
    if not interview_session['active']:
        return
    
    current_phase = interview_session['phase']
    
    if current_phase == "thinking":
        print(f"💭 생각 시간 종료! 답변 시작 (Q{interview_session['question_index']})")
        interview_session['phase'] = "answering"
        interview_session['phase_start_time'] = time.time()
        
        interview_session['auto_timer'] = threading.Timer(
            interview_session['answer_time'], 
            auto_next_phase
        )
        interview_session['auto_timer'].start()
        
    elif current_phase == "answering":
        print(f"⏰ 답변 시간 종료! 다음 질문으로 이동")
        next_question_auto()

def next_question_auto():
    global interview_session
    
    if interview_session['question_index'] < len(interview_session['questions_list']):
        current_q = interview_session['questions_list'][interview_session['question_index']]
        
        interview_session.update({
            'question_index': interview_session['question_index'] + 1,
            'current_question': current_q['question'],
            'category': current_q['category'],
            'phase': 'thinking',
            'phase_start_time': time.time(),
            'current_flow_stage': current_q.get('stage', 0)
        })
        
        print(f"📝 {current_q['category']} 단계: {current_q['question'][:50]}...")
        
        interview_session['auto_timer'] = threading.Timer(
            interview_session['thinking_time'], 
            auto_next_phase
        )
        interview_session['auto_timer'].start()
        
    else:
        print("🎉 모든 질문이 완료되었습니다!")
        interview_session['phase'] = 'finished'
        
        voice_analysis_data["session_active"] = False
        
        # AI 종합 분석 실행
        threading.Thread(target=run_comprehensive_ai_analysis, daemon=True).start()
        
        threading.Timer(3.0, stop_interview_internal).start()

def run_comprehensive_ai_analysis():
    """면접 종료 후 AI 종합 분석"""
    try:
        print("🤖 AI 종합 분석 시작...")
        
        # 저장된 최종 오디오 파일 확인
        audio_file = "final_interview_audio.webm"
        
        if not os.path.exists(audio_file):
            print(f"⚠️ 오디오 파일 없음: {audio_file}")
            # 기본 리포트 생성
            generate_voice_analysis_report()
            return
        
        # librosa 음정 떨림 분석
        tremor_analysis = None
        if LIBROSA_AVAILABLE:
            tremor_analysis = analyze_voice_tremor_librosa(audio_file)
            if tremor_analysis:
                voice_analysis_data["ai_analysis"]["tremor"] = tremor_analysis
        
        # Whisper + KoNLPy 텍스트 분석
        speech_analysis = None
        if WHISPER_AVAILABLE:
            speech_analysis = analyze_speech_with_whisper_konlpy(audio_file)
            if speech_analysis:
                voice_analysis_data["ai_analysis"]["speech"] = speech_analysis
        
        # 최종 리포트 생성
        generate_voice_analysis_report()
        
        print("✅ AI 종합 분석 완료")
        
    except Exception as e:
        print(f"❌ AI 종합 분석 오류: {e}")
        import traceback
        traceback.print_exc()
        # 오류 시에도 기본 리포트는 생성
        generate_voice_analysis_report()

def stop_interview_internal():
    """내부적으로 면접을 중지하는 함수"""
    global interview_session
    
    if interview_session.get('auto_timer'):
        interview_session['auto_timer'].cancel()
    
    interview_session.update({
        'active': False,
        'current_question': None,
        'phase': 'finished',
        'auto_timer': None,
        'current_flow_stage': 0
    })
    
    voice_analysis_data["session_active"] = False
    
    print("🎬 면접 자동 종료 완료")

# ==================== 자세 분석기 ====================

class PrecisePostureAnalyzer:
    def __init__(self):
        self.guide_frame = {
            'x_min': 0.25, 'x_max': 0.75,
            'y_min': 0.15, 'y_max': 0.85
        }
        self.calibrated = False
        self.calibration_samples = []
        self.thresholds = {
            'head_rotation': 0.03, 'head_tilt': 0.02,
            'shoulder_level': 0.05, 'movement': 0.03,
            'face_touch': 0.12, 'arm_cross': 0.15, 'slouch': 0.13
        }
        self.previous_positions = deque(maxlen=15)
        self.last_beep_time = 0
        self.beep_cooldown = 2.0

    def add_calibration_sample(self, landmarks):
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        nose_to_shoulder_y = nose.y - shoulder_center_y
        self.calibration_samples.append({
            'nose_to_shoulder_y': nose_to_shoulder_y,
            'nose_x': nose.x
        })

    def finalize_calibration(self):
        if len(self.calibration_samples) >= 10:
            avg_nose_to_shoulder_y = sum(s['nose_to_shoulder_y'] for s in self.calibration_samples) / len(self.calibration_samples)
            avg_nose_x = sum(s['nose_x'] for s in self.calibration_samples) / len(self.calibration_samples)
            self.baseline_nose_to_shoulder_y = avg_nose_to_shoulder_y
            self.baseline_nose_x = avg_nose_x
            self.calibrated = True
            print(f"✅ 자세 캘리브레이션 완료!")
            return True
        return False

    def reset_calibration(self):
        self.calibrated = False
        self.calibration_samples = []

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def check_frame_position(self, landmarks):
        global is_analysis_active
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        margin = 0.1
        x_min_with_margin = self.guide_frame['x_min'] + margin
        x_max_with_margin = self.guide_frame['x_max'] - margin
        y_min_with_margin = self.guide_frame['y_min'] + margin
        y_max_with_margin = self.guide_frame['y_max'] - margin

        out_of_frame = (
            nose.x < x_min_with_margin or nose.x > x_max_with_margin or
            nose.y < y_min_with_margin or nose.y > y_max_with_margin
        )

        if out_of_frame and is_analysis_active:
            current_time = time.time()
            if current_time - self.last_beep_time > self.beep_cooldown:
                threading.Thread(target=play_beep, daemon=True).start()
                self.last_beep_time = current_time

        return not out_of_frame

    def analyze_face_direction(self, landmarks):
        issues = []
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        current_time = time.time()
        eye_distance = abs(left_eye.x - right_eye.x)
        eye_height_diff = left_eye.y - right_eye.y
        
        normalized_tilt = eye_height_diff / eye_distance if eye_distance >= 0.05 else 0
        
        self.previous_positions.append({
            'nose_x': nose.x, 'nose_y': nose.y,
            'tilt': normalized_tilt, 'timestamp': current_time
        })

        if len(self.previous_positions) >= 8:
            recent_positions = list(self.previous_positions)[-8:]
            avg_x = np.mean([p['nose_x'] for p in recent_positions])
            std_x = np.std([p['nose_x'] for p in recent_positions])
            deviation = abs(nose.x - avg_x)
            
            if deviation > 0.03 and std_x > 0.02:
                issues.append("고개를 좌우로 흔들지 마세요")

        if len(self.previous_positions) >= 5:
            recent_tilts = [p['tilt'] for p in list(self.previous_positions)[-5:]]
            avg_tilt = np.mean(recent_tilts)
            
            if avg_tilt > 0.25:
                issues.append("고개가 오른쪽으로 기울어져 있습니다")
            elif avg_tilt < -0.25:
                issues.append("고개가 왼쪽으로 기울어져 있습니다")

        return issues, {'eye_tilt': float(normalized_tilt), 'face_rotation': float(nose.x)}

    def analyze_shoulder_posture(self, landmarks):
        issues = []
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_diff = left_shoulder.y - right_shoulder.y
        if shoulder_diff > 0.07:
            issues.append("왼쪽 어깨가 올라가 있습니다")
        elif shoulder_diff < -0.07:
            issues.append("오른쪽 어깨가 올라가 있습니다")

        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        if shoulder_center_y > hip_center_y + 0.13:
            issues.append("등을 펴고 바르게 앉으세요")

        return issues

    def analyze_hand_gestures(self, landmarks):
        issues = []
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        if self.calculate_distance(left_wrist, nose) < self.thresholds['face_touch']:
            issues.append("왼손이 얼굴 근처에 있습니다")
        if self.calculate_distance(right_wrist, nose) < self.thresholds['face_touch']:
            issues.append("오른손이 얼굴 근처에 있습니다")

        wrist_distance = self.calculate_distance(left_wrist, right_wrist)
        avg_wrist_y = (left_wrist.y + right_wrist.y) / 2
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        if wrist_distance < self.thresholds['arm_cross'] and avg_wrist_y > avg_shoulder_y:
            issues.append("팔짱을 끼지 마세요")

        return issues

    def analyze_movement(self):
        if len(self.previous_positions) < 2:
            return [], 0.0

        recent_positions = list(self.previous_positions)[-5:]
        if len(recent_positions) < 2:
            return [], 0.0

        movements = []
        for i in range(1, len(recent_positions)):
            prev = recent_positions[i-1]
            curr = recent_positions[i]
            movement = np.sqrt(
                (curr['nose_x'] - prev['nose_x']) ** 2 +
                (curr['nose_y'] - prev['nose_y']) ** 2
            )
            movements.append(movement)

        avg_movement = np.mean(movements) if movements else 0.0
        issues = []
        if avg_movement > self.thresholds['movement']:
            issues.append("몸이 많이 흔들립니다")

        return issues, float(avg_movement)

    def analyze_posture(self, landmarks):
        issues = []
        score = 100

        in_frame = self.check_frame_position(landmarks)
        if not in_frame:
            issues.append("가이드 영역 안으로 이동하세요")
            score -= 30

        face_issues, _ = self.analyze_face_direction(landmarks)
        issues.extend(face_issues)
        score -= len(face_issues) * 15

        shoulder_issues = self.analyze_shoulder_posture(landmarks)
        issues.extend(shoulder_issues)
        score -= len(shoulder_issues) * 15

        hand_issues = self.analyze_hand_gestures(landmarks)
        issues.extend(hand_issues)
        score -= len(hand_issues) * 10

        movement_issues, movement_val = self.analyze_movement()
        issues.extend(movement_issues)
        score -= len(movement_issues) * 10

        return {
            'score': max(0, score),
            'issues': issues,
            'in_frame': in_frame,
            'movement': movement_val
        }

analyzer = PrecisePostureAnalyzer()

latest_analysis_data = {
    'score': 100, 'issues': [], 'timestamp': time.time(),
    'interview': {
        'active': False, 'current_question': None,
        'phase': 'thinking', 'time_remaining': 0
    }
}

is_analysis_active = False
calibration_start_time = None
calibration_in_progress = False

# ==================== 비디오 그리기 함수들 ====================

def draw_fixed_guide_frame(frame, h, w):
    guide = analyzer.guide_frame
    x1, y1 = int(w * guide['x_min']), int(h * guide['y_min'])
    x2, y2 = int(w * guide['x_max']), int(h * guide['y_max'])

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, y1), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, y2), (w, h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, y1), (x1, y2), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x2, y1), (w, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    color = (0, 255, 0)
    thickness = 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)



def generate_frames():
    global latest_analysis_data, is_analysis_active, calibration_start_time, calibration_in_progress

    while True:
        success, frame = camera.read()
        if not success:
            print("⚠️ 카메라에서 프레임을 읽을 수 없습니다.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        draw_fixed_guide_frame(frame, h, w)


        if pose:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)

            if pose_results.pose_landmarks:
                try:
                    if calibration_in_progress:
                        elapsed = time.time() - calibration_start_time
                        if elapsed < 5.0:
                            analyzer.add_calibration_sample(pose_results.pose_landmarks.landmark)
                            countdown = 5 - int(elapsed)
                            cv2.putText(frame, f"Calibrating... {countdown}", (w // 2 - 150, h // 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                            cv2.putText(frame, "Keep current posture", (w // 2 - 150, h // 2 + 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        else:
                            analyzer.finalize_calibration()
                            calibration_in_progress = False
                            is_analysis_active = True

                    elif analyzer.calibrated and is_analysis_active:
                        analysis = analyzer.analyze_posture(pose_results.pose_landmarks.landmark)

                        current_time = time.time()
                        time_remaining = 0
                        if interview_session['phase_start_time']:
                            elapsed = int(current_time - interview_session['phase_start_time'])
                            if interview_session['phase'] == 'thinking':
                                time_remaining = max(0, interview_session['thinking_time'] - elapsed)
                            elif interview_session['phase'] == 'answering':
                                time_remaining = max(0, interview_session['answer_time'] - elapsed)

                        interview_info = {
                            'active': interview_session["active"],
                            'current_question': interview_session["current_question"],
                            'category': interview_session["category"],
                            'question_number': interview_session["question_index"],
                            'total_questions': len(interview_session["questions_list"]),
                            'phase': interview_session["phase"],
                            'time_remaining': time_remaining,
                            'current_stage': INTERVIEW_FLOW[interview_session.get('current_flow_stage', 0)]
                        }

                        latest_analysis_data = {
                            'score': analysis['score'],
                            'issues': analysis['issues'],
                            'timestamp': time.time(),
                            'interview': interview_info
                        }

                        if not interview_session["active"]:
                            y_offset = 30
                            score_color = (0, 255, 0) if analysis['score'] >= 80 else (0, 165, 255) if analysis['score'] >= 60 else (0, 0, 255)
                            cv2.putText(frame, f"Score: {analysis['score']}", (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)
                            y_offset += 35

                            if analysis['issues']:
                                for issue in analysis['issues']:
                                    cv2.putText(frame, issue, (10, y_offset),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    y_offset += 25
                            else:
                                cv2.putText(frame, "Perfect!", (10, y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Waiting for calibration...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                except Exception as e:
                    print(f"분석 오류: {e}")
            else:
                cv2.putText(frame, "No pose detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_data_stream():
    while True:
        current_time = time.time()
        time_remaining = 0
        
        if interview_session['phase_start_time']:
            elapsed = int(current_time - interview_session['phase_start_time'])
            if interview_session['phase'] == 'thinking':
                time_remaining = max(0, interview_session['thinking_time'] - elapsed)
            elif interview_session['phase'] == 'answering':
                time_remaining = max(0, interview_session['answer_time'] - elapsed)

        interview_info = {
            'active': interview_session['active'],
            'current_question': interview_session['current_question'],
            'category': interview_session['category'],
            'question_number': interview_session['question_index'],
            'total_questions': len(interview_session['questions_list']),
            'phase': interview_session['phase'],
            'time_remaining': time_remaining,
            'current_stage': INTERVIEW_FLOW[interview_session.get('current_flow_stage', 0)] if interview_session['active'] else None
        }

        voice_info = {
            'session_active': voice_analysis_data["session_active"],
            'analysis_complete': voice_analysis_data["analysis_complete"],
            'final_report': voice_analysis_data["final_report"] if voice_analysis_data["analysis_complete"] else None,
            'has_data': len(voice_analysis_data["audio_samples"]) > 0,
            'sample_count': len(voice_analysis_data["audio_samples"]),
            'filler_count': len(voice_analysis_data["filler_words"]),
            'chunk_count': voice_analysis_data["chunk_count"],
            'ai_enabled': WHISPER_AVAILABLE and LIBROSA_AVAILABLE
        }

        stream_data = {
            **latest_analysis_data,
            'interview': interview_info,
            'voice_analysis': voice_info
        }

        yield f"data: {json.dumps(stream_data)}\n\n"
        time.sleep(0.5)

# ==================== API 엔드포인트들 ====================

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok', 
        'camera': camera.isOpened() if camera else False,
        'voice_analysis': 'AI-Powered' if (WHISPER_AVAILABLE and LIBROSA_AVAILABLE) else 'Basic',
        'ai_modules': {
            'whisper': WHISPER_AVAILABLE,
            'konlpy': KONLPY_AVAILABLE,
            'librosa': LIBROSA_AVAILABLE,
            'pydub': AUDIO_PROCESSING_AVAILABLE
        },
        'resolution': f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}" if camera else "N/A"
    })

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_data')
def stream_data():
    return Response(generate_data_stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    global is_analysis_active, calibration_start_time, calibration_in_progress
    analyzer.reset_calibration()
    calibration_start_time = time.time()
    calibration_in_progress = True
    is_analysis_active = False
    print("🎯 자세 분석 시작 - 캘리브레이션 중...")
    return jsonify({'status': 'calibration_started'})

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    global is_analysis_active, calibration_in_progress
    is_analysis_active = False
    calibration_in_progress = False
    analyzer.last_beep_time = 0
    
    if interview_session.get('auto_timer'):
        interview_session['auto_timer'].cancel()
    
    interview_session.update({
        'active': False, 
        'current_question': None,
        'phase': 'finished', 
        'auto_timer': None
    })
    
    # 음성 분석 데이터 초기화
    voice_analysis_data["session_active"] = False
    voice_analysis_data["analysis_complete"] = False
    voice_analysis_data["final_report"] = {}
    voice_analysis_data["ai_analysis"] = {}
    
    print("🛑 자세 분석 중지")
    return jsonify({'status': 'stopped'})

@app.route('/interview/start', methods=['POST'])
def start_interview():
    global interview_session
    
    if interview_session.get('auto_timer'):
        interview_session['auto_timer'].cancel()
    
    structured_questions = generate_structured_interview_questions()
    first_q = structured_questions[0]
    
    interview_session.update({
        'active': True,
        'questions_list': structured_questions,
        'question_index': 1,
        'current_question': first_q['question'],
        'category': first_q['category'],
        'phase': 'thinking',
        'phase_start_time': time.time(),
        'current_flow_stage': first_q['stage']
    })
    
    # 음성 분석 데이터 완전 초기화
    voice_analysis_data.clear()
    voice_analysis_data.update({
        "session_active": True,
        "start_time": time.time(),
        "audio_samples": [],
        "filler_words": [],
        "voice_confidence": [],
        "speaking_pace": [],
        "volume_levels": [],
        "total_speaking_time": 0,
        "silence_periods": [],
        "analysis_complete": False,
        "final_report": {},
        "chunk_count": 0,
        "ai_analysis": {}
    })
    
    print(f"🎬 체계적 면접 시작! 총 {len(structured_questions)}개 질문")
    print(f"🎤 AI 음성 분석 세션 활성화 (Whisper: {WHISPER_AVAILABLE}, librosa: {LIBROSA_AVAILABLE})")
    
    interview_session['auto_timer'] = threading.Timer(interview_session['thinking_time'], auto_next_phase)
    interview_session['auto_timer'].start()
    
    return jsonify({
        'status': 'started',
        'total_questions': len(structured_questions),
        'current_question': first_q['question'],
        'category': first_q['category'],
        'flow_stage': INTERVIEW_FLOW[first_q['stage']],
        'voice_analysis_active': True,
        'ai_enabled': WHISPER_AVAILABLE and LIBROSA_AVAILABLE
    })

@app.route('/interview/stop', methods=['POST'])
def stop_interview():
    """HTTP 엔드포인트로서의 면접 중지 함수"""
    stop_interview_internal()
    return jsonify({'status': 'stopped'})

# MediaRecorder 오디오 청크 수신
@app.route('/voice/audio_chunk_blob', methods=['POST'])
def receive_audio_chunk_blob():
    """MediaRecorder에서 생성된 오디오 청크 수신 및 분석"""
    try:
        if not voice_analysis_data["session_active"]:
            return jsonify({'status': 'session_inactive'})
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'})
        
        audio_file = request.files['audio']
        chunk_number = int(request.form.get('chunk_number', 0))
        
        audio_data = audio_file.read()
        voice_analysis_data["chunk_count"] += 1
        
        # 오디오 청크 저장 (나중에 병합)
        chunk_path = f"temp_chunk_{chunk_number}.webm"
        with open(chunk_path, 'wb') as f:
            f.write(audio_data)
        
        print(f"📥 청크 #{chunk_number} 수신: {len(audio_data)} bytes")
        
        # 기본 분석
        if len(audio_data) > 100:
            basic_volume = min(100, (len(audio_data) / 1000) * 8)
            
            basic_sample = {
                'volume': basic_volume / 100,
                'is_speaking': True,
                'fundamental_freq': 0,
                'voice_tremor': 0,
                'timestamp': time.time(),
                'analysis_type': 'basic'
            }
            
            voice_analysis_data["audio_samples"].append(basic_sample)
        
        # pydub로 정밀 분석 시도
        if AUDIO_PROCESSING_AVAILABLE:
            try:
                audio_segment = AudioSegment.from_file(
                    io.BytesIO(audio_data),
                    format="webm"
                )
                
                samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                
                if audio_segment.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                
                if len(samples) > 0 and np.max(np.abs(samples)) > 0:
                    samples = samples / np.max(np.abs(samples))
                    
                    precise_analysis = analyze_audio_chunk(samples, audio_segment.frame_rate)
                    
                    if precise_analysis:
                        voice_analysis_data["audio_samples"][-1] = precise_analysis
                        voice_analysis_data["audio_samples"][-1]['analysis_type'] = 'precise'
                
            except Exception as e:
                print(f"⚠️ 정밀 분석 실패 (기본 분석 유지): {e}")
        
        return jsonify({
            'status': 'success',
            'chunk_number': chunk_number,
            'samples_total': len(voice_analysis_data["audio_samples"]),
            'chunks_received': voice_analysis_data["chunk_count"]
        })
        
    except Exception as e:
        print(f"❌ 청크 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

# 최종 오디오 파일 수신
@app.route('/voice/final_audio', methods=['POST'])
def receive_final_audio():
    """최종 완성된 오디오 파일 수신"""
    try:
        if 'final_audio' not in request.files:
            return jsonify({'error': 'No final audio file'})
        
        audio_file = request.files['final_audio']
        timestamp = request.form.get('timestamp', time.time() * 1000)
        
        # 최종 오디오 저장
        final_audio_path = "final_interview_audio.webm"
        audio_file.save(final_audio_path)
        
        file_size = os.path.getsize(final_audio_path)
        print(f"📥 최종 오디오 파일 수신 - 크기: {file_size} bytes")
        
        # AI 분석 시작 (백그라운드)
        if WHISPER_AVAILABLE or LIBROSA_AVAILABLE:
            print("🤖 AI 분석 스레드 시작...")
            threading.Thread(target=run_comprehensive_ai_analysis, daemon=True).start()
        
        return jsonify({
            'status': 'final_audio_received',
            'file_size': file_size,
            'timestamp': timestamp,
            'ai_analysis_started': WHISPER_AVAILABLE or LIBROSA_AVAILABLE
        })
        
    except Exception as e:
        print(f"❌ 최종 오디오 파일 처리 실패: {e}")
        return jsonify({'error': str(e)})

# 음성 인식 텍스트 수신
@app.route('/voice/speech_text', methods=['POST'])
def receive_speech_text():
    """음성 인식 텍스트 수신 및 분석"""
    try:
        if not voice_analysis_data["session_active"]:
            return jsonify({'status': 'session_inactive'})
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text data'})
        
        text = data['text']
        
        # 추임새 단어 기록
        words = text.split()
        for word in words:
            clean_word = word.strip('.,!?').lower()
            if clean_word in FILLER_WORDS:
                voice_analysis_data["filler_words"].append(clean_word)
        
        print(f"📝 음성 텍스트 분석: {text[:50]}...")
        
        return jsonify({
            'status': 'processed',
            'total_fillers': len(voice_analysis_data["filler_words"])
        })
        
    except Exception as e:
        print(f"음성 텍스트 처리 오류: {e}")
        return jsonify({'error': str(e)})

@app.route('/voice/analysis_report', methods=['GET'])
def get_voice_analysis_report():
    """음성 분석 리포트 조회"""
    if voice_analysis_data["analysis_complete"]:
        return jsonify(voice_analysis_data["final_report"])
    else:
        return jsonify({
            'status': 'analysis_in_progress',
            'ai_enabled': WHISPER_AVAILABLE and LIBROSA_AVAILABLE,
            'samples': len(voice_analysis_data["audio_samples"])
        })

# ==================== 리소스 정리 ====================

def safe_close_mediapipe():
    """MediaPipe 리소스를 안전하게 종료"""
    global pose
    
    if pose is None:
        return
    
    try:
        if hasattr(pose, '_graph'):
            if pose._graph is not None:
                pose.close()
                print("✅ MediaPipe 정상 종료")
            else:
                print("ℹ️ MediaPipe 이미 종료됨")
        else:
            try:
                pose.close()
                print("✅ MediaPipe 정상 종료 (구버전)")
            except:
                print("ℹ️ MediaPipe 종료 시도 완료")
                
    except ValueError as e:
        if "already None" in str(e) or "Closing" in str(e):
            print("ℹ️ MediaPipe 이미 종료된 상태")
        else:
            print(f"⚠️ MediaPipe 종료 중 ValueError: {e}")
    except Exception as e:
        print(f"❌ MediaPipe 종료 중 오류: {e}")
    finally:
        pose = None

def cleanup_resources():
    """프로그램 종료 시 모든 리소스를 안전하게 정리"""
    print("🧹 리소스 정리 시작...")
    
    global camera
    
    # 카메라 리소스 정리
    try:
        if camera is not None:
            if camera.isOpened():
                camera.release()
                print("✅ 카메라 리소스 해제 완료")
            camera = None
    except Exception as e:
        print(f"❌ 카메라 정리 중 오류: {e}")
    
    # MediaPipe 안전 종료
    safe_close_mediapipe()
    
    # 면접 세션 정리
    try:
        global interview_session
        if interview_session.get('auto_timer'):
            interview_session['auto_timer'].cancel()
            print("✅ 면접 타이머 정리 완료")
    except Exception as e:
        print(f"❌ 면접 세션 정리 중 오류: {e}")
    
    # 음성 분석 데이터 정리
    try:
        global voice_analysis_data
        voice_analysis_data["session_active"] = False
        voice_analysis_data["audio_samples"].clear()
        voice_analysis_data["filler_words"].clear()
        print("✅ 음성 분석 데이터 정리 완료")
    except Exception as e:
        print(f"❌ 음성 분석 데이터 정리 중 오류: {e}")
    
    # 임시 오디오 파일 정리
    try:
        for f in os.listdir('.'):
            if f.startswith('temp_chunk_') or f == 'final_interview_audio.webm':
                os.remove(f)
                print(f"✅ 임시 파일 삭제: {f}")
    except Exception as e:
        print(f"⚠️ 임시 파일 정리 중 오류: {e}")
    
    print("🧹 리소스 정리 완료")

def safe_shutdown():
    """프로그램 안전 종료"""
    print("\n🛑 안전한 서버 종료 시작...")
    
    try:
        global interview_session
        if interview_session.get('active'):
            print("📝 진행 중인 면접 종료 중...")
            stop_interview_internal()
    except Exception as e:
        print(f"❌ 면접 종료 중 오류: {e}")
    
    try:
        cleanup_resources()
    except Exception as e:
        print(f"❌ 리소스 정리 중 오류: {e}")
    
    print("✅ 서버 종료 완료")

# atexit 등록
atexit.register(cleanup_resources)

# ==================== 메인 실행 ====================

if __name__ == '__main__':
    print("=" * 90)
    print("🎓 AI 기반 체육대학 면접 분석 시스템 (Whisper + KoNLPy + librosa)")
    print("=" * 90)
    print(f"🌐 서버 주소: http://0.0.0.0:5001")
    print(f"📱 카메라: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}" if camera else "N/A")
    print("🎯 면접 시스템: 체계적 흐름 (자기소개 → 지원동기 → 전공지식 → 상황대처 → 압박질문 → 마무리)")
    print("\n🤖 AI 모듈 상태:")
    print(f"   - Whisper (음성인식): {'✅ 활성화' if WHISPER_AVAILABLE else '❌ 비활성화'}")
    print(f"   - KoNLPy (품사분석): {'✅ 활성화' if KONLPY_AVAILABLE else '❌ 비활성화'}")
    print(f"   - librosa (음정떨림): {'✅ 활성화' if LIBROSA_AVAILABLE else '❌ 비활성화'}")
    print(f"   - pydub (오디오처리): {'✅ 활성화' if AUDIO_PROCESSING_AVAILABLE else '❌ 비활성화'}")
    
    if WHISPER_AVAILABLE and LIBROSA_AVAILABLE:
        print("\n📊 AI 분석 항목:")
        print("   - 음정 떨림 (Jitter/Shimmer)")
        print("   - 추임새 사용 빈도 및 종류")
        print("   - 접속사 사용 패턴")
        print("   - 반복 단어 감지")
        print("   - 품사별 분석 (명사, 동사, 형용사)")
        print("   - 어휘 다양성")
        print("   - 문장 길이 분석")
    else:
        print("\n⚠️ AI 모듈 미설치:")
        if not WHISPER_AVAILABLE:
            print("   pip install openai-whisper")
        if not KONLPY_AVAILABLE:
            print("   pip install konlpy")
        if not LIBROSA_AVAILABLE:
            print("   pip install librosa soundfile")
    
    print("=" * 90)

    try:
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C 감지 - 서버 종료 중...")
        safe_shutdown()
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
        safe_shutdown()
    finally:
        try:
            safe_shutdown()
        except:
            pass
        print("🏁 프로그램 완전 종료")