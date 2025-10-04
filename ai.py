# server_webrtc.py - AWS EC2용 WebRTC 기반 면접 분석 시스템
# 클라이언트 브라우저에서 카메라 영상을 받아서 분석

from flask import Flask, jsonify, request
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import cv2
import json
import time
from collections import deque, Counter
import threading
import random
import atexit
import io
import os
import base64

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
    print("🔄 Whisper 모델 로딩 중...")
    whisper_model = whisper.load_model("tiny")
    print("✅ Whisper 모델 로드 완료")

if KONLPY_AVAILABLE:
    okt = Okt()
    print("✅ KoNLPy Okt 초기화 완료")

# 기존 코드의 모든 전역 변수들 (voice_analysis_data, FILLER_WORDS, INTERVIEW_QUESTIONS 등)
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
    "ai_analysis": {}
}

FILLER_WORDS = [
    "어", "음", "아", "그", "뭐", "이제", "그래서", "그니까", "그런데",
    "어떻게", "뭔가", "약간", "좀", "그거", "이거", "저기", "아무튼",
    "일단", "우선", "그러면", "그럼", "아니", "맞아", "그치", "응", "네",
    "에", "엄", "흠", "아무래도"
]

CONNECTIVES = [
    "그래서", "그러나", "하지만", "그리고", "또한", "따라서", "그런데",
    "왜냐하면", "즉", "결국", "그러므로", "그렇지만", "또는", "혹은",
    "그럼에도", "반면에", "한편", "더불어"
]

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

interview_session = {
    "active": False,
    "current_question": None,
    "question_index": 0,
    "category": None,
    "questions_list": [],
    "start_time": None,
    "thinking_time": 10,
    "answer_time": 50,
    "phase": "thinking",
    "auto_timer": None,
    "phase_start_time": None,
    "current_flow_stage": 0
}

# 자세 분석기 (기존 코드)
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

        return not out_of_frame

    def analyze_face_direction(self, landmarks):
        issues = []
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        eye_distance = abs(left_eye.x - right_eye.x)
        eye_height_diff = left_eye.y - right_eye.y
        
        normalized_tilt = eye_height_diff / eye_distance if eye_distance >= 0.05 else 0
        
        self.previous_positions.append({
            'nose_x': nose.x, 'nose_y': nose.y,
            'tilt': normalized_tilt, 'timestamp': time.time()
        })

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

        if self.calculate_distance(left_wrist, nose) < self.thresholds['face_touch']:
            issues.append("왼손이 얼굴 근처에 있습니다")
        if self.calculate_distance(right_wrist, nose) < self.thresholds['face_touch']:
            issues.append("오른손이 얼굴 근처에 있습니다")

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
calibration_in_progress = False

# AI 분석 함수들 (기존 코드 그대로)
def analyze_voice_tremor_librosa(audio_path):
    if not LIBROSA_AVAILABLE:
        return None
    
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) < sr * 0.5:
            return None
        
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
        
        rms = librosa.feature.rms(y=y)[0]
        shimmer = np.std(rms)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_stability = np.std(spectral_centroid)
        
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
        
        return result
        
    except Exception as e:
        print(f"❌ librosa 분석 실패: {e}")
        return None

def analyze_speech_with_whisper_konlpy(audio_path):
    if not WHISPER_AVAILABLE:
        return None
    
    try:
        result = whisper_model.transcribe(audio_path, language='ko')
        full_text = result['text']
        
        if not full_text.strip():
            return None
        
        words = full_text.split()
        
        filler_count = Counter()
        for word in words:
            clean_word = word.strip('.,!?').lower()
            if clean_word in FILLER_WORDS:
                filler_count[clean_word] += 1
        
        connective_count = Counter()
        for word in words:
            clean_word = word.strip('.,!?').lower()
            if clean_word in CONNECTIVES:
                connective_count[clean_word] += 1
        
        nouns = []
        verbs = []
        adjectives = []
        adverbs = []
        
        if KONLPY_AVAILABLE and okt:
            try:
                morphs = okt.pos(full_text)
                nouns = [word for word, pos in morphs if pos == 'Noun']
                verbs = [word for word, pos in morphs if pos == 'Verb']
                adjectives = [word for word, pos in morphs if pos == 'Adjective']
                adverbs = [word for word, pos in morphs if pos == 'Adverb']
            except:
                pass
        
        word_freq = Counter(word.strip('.,!?').lower() for word in words if len(word) > 1)
        repeated_words = {word: count for word, count in word_freq.items() if count > 2}
        
        sentences = [s.strip() for s in full_text.split('.') if s.strip()]
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        vocabulary_richness = len(set(nouns)) / len(nouns) if nouns else 0
        
        analysis_result = {
            'full_text': full_text,
            'total_words': len(words),
            'filler_words': dict(filler_count),
            'filler_total': sum(filler_count.values()),
            'filler_ratio': (sum(filler_count.values()) / len(words) * 100) if words else 0,
            'connectives': dict(connective_count),
            'connective_total': sum(connective_count.values()),
            'noun_count': len(nouns),
            'verb_count': len(verbs),
            'adjective_count': len(adjectives),
            'adverb_count': len(adverbs),
            'top_nouns': dict(Counter(nouns).most_common(10)),
            'repeated_words': repeated_words,
            'sentence_count': len(sentences),
            'avg_sentence_length': avg_sentence_length,
            'vocabulary_richness': vocabulary_richness * 100,
            'segments': result.get('segments', [])
        }
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ Whisper+KoNLPy 분석 실패: {e}")
        return None

def generate_ai_recommendations(tremor, speech):
    recommendations = []
    
    if tremor:
        if tremor['jitter_percent'] > 3.0:
            recommendations.append(f"목소리에 떨림이 있습니다 (Jitter: {tremor['jitter_percent']:.1f}%). 심호흡을 하고 천천히 말해보세요.")
        elif tremor['jitter_percent'] > 2.0:
            recommendations.append("약간의 긴장이 느껴집니다. 편안하게 답변해보세요.")
        
        if tremor['shimmer'] > 0.05:
            recommendations.append("음량 변화가 큽니다. 일정한 크기로 말해보세요.")
    
    if speech:
        if speech['filler_ratio'] > 10:
            recommendations.append(f"추임새가 많습니다 ({speech['filler_total']}회, {speech['filler_ratio']:.1f}%). 의식적으로 줄여보세요.")
        
        if speech['connective_total'] > 15:
            recommendations.append(f"접속사 사용이 많습니다 ({speech['connective_total']}회). 문장을 간결하게 만들어보세요.")
        
        if speech.get('repeated_words'):
            top_repeated = sorted(speech['repeated_words'].items(), key=lambda x: x[1], reverse=True)[:3]
            if top_repeated:
                recommendations.append(f"반복된 단어: {', '.join([f'{w}({c}회)' for w, c in top_repeated])}")
        
        if speech['vocabulary_richness'] < 30:
            recommendations.append("다양한 어휘를 사용해보세요.")
    
    if not recommendations:
        recommendations.append("훌륭한 답변입니다!")
    
    return recommendations

def generate_voice_analysis_report():
    """음성 분석 리포트 생성 - 타임스탬프 필수 포함"""
    try:
        print("=" * 70)
        print("📊 음성 분석 리포트 생성 시작")
        print("=" * 70)
        
        sample_count = len(voice_analysis_data["audio_samples"])
        print(f"📦 분석 샘플: {sample_count}개")
        print(f"📦 수신 청크: {voice_analysis_data.get('chunk_count', 0)}개")
        
        # 기본 음성 분석
        volumes = [sample['volume'] for sample in voice_analysis_data["audio_samples"] if sample]
        speaking_periods = [sample for sample in voice_analysis_data["audio_samples"] 
                          if sample and sample.get('is_speaking', False)]
        
        total_time = time.time() - voice_analysis_data["start_time"] if voice_analysis_data["start_time"] else 0
        speaking_time = len(speaking_periods) * 0.1
        speaking_ratio = speaking_time / total_time if total_time > 0 else 0
        
        avg_volume = np.mean(volumes) if volumes else 0
        volume_std = np.std(volumes) if len(volumes) > 1 else 0
        volume_consistency = max(0, 1 - (volume_std / (avg_volume + 0.001)))
        
        # 기본 신뢰도 점수 계산
        confidence_score = (
            (min(avg_volume * 100, 100) * 0.30) +
            (volume_consistency * 100 * 0.25) +
            (speaking_ratio * 100 * 0.20)
        )
        
        print(f"📊 기본 분석 결과:")
        print(f"   - 평균 볼륨: {avg_volume * 100:.1f}%")
        print(f"   - 볼륨 일관성: {volume_consistency * 100:.1f}%")
        print(f"   - 말하기 비율: {speaking_ratio * 100:.1f}%")
        print(f"   - 기본 신뢰도: {confidence_score:.1f}")
        
        # AI 분석 결과 반영
        ai_analysis = voice_analysis_data.get("ai_analysis", {})
        
        if ai_analysis:
            print("🤖 AI 분석 결과 반영 중...")
            
            if ai_analysis.get('tremor'):
                tremor_confidence = ai_analysis['tremor']['voice_confidence']
                confidence_score = (confidence_score * 0.5) + (tremor_confidence * 0.5)
                print(f"   ✅ librosa 분석: Jitter={ai_analysis['tremor']['jitter_percent']:.2f}%")
            
            if ai_analysis.get('speech'):
                filler_penalty = min(ai_analysis['speech']['filler_ratio'], 20)
                confidence_score = confidence_score * (1 - filler_penalty / 100)
                print(f"   ✅ Whisper 분석: 추임새={ai_analysis['speech']['filler_total']}회")
        
        # 추천사항 생성
        recommendations = []
        
        if avg_volume < 0.3:
            recommendations.append("목소리가 작습니다. 더 자신있게 말해보세요.")
        
        if ai_analysis:
            ai_recommendations = generate_ai_recommendations(
                ai_analysis.get('tremor'),
                ai_analysis.get('speech')
            )
            recommendations.extend(ai_recommendations)
        
        if not recommendations:
            recommendations.append("전반적으로 안정적인 음성으로 답변하셨습니다!")
        
        # 타임스탬프 생성 (필수!)
        current_timestamp = time.time()
        
        # 리포트 구성
        report = {
            "overall_score": round(confidence_score, 1),
            "analysis_timestamp": current_timestamp,  # 필수!
            "detailed_analysis": {
                "voice_confidence": round(confidence_score, 1),
                "average_volume": round(avg_volume * 100, 1),
                "volume_consistency": round(volume_consistency * 100, 1),
                "voice_stability": round(volume_consistency * 100, 1),
                "speaking_ratio": round(speaking_ratio * 100, 1),
                "total_speaking_time": round(speaking_time, 1),
                "filler_word_count": 0,
                "filler_ratio": 0
            },
            "recommendations": recommendations[:10],
            "ai_powered": bool(ai_analysis),
            "debug_info": {
                "samples_analyzed": sample_count,
                "chunks_received": voice_analysis_data.get("chunk_count", 0),
                "has_whisper": WHISPER_AVAILABLE,
                "has_librosa": LIBROSA_AVAILABLE
            }
        }

        # AI 세부 정보 추가
        if ai_analysis.get('speech'):
            speech = ai_analysis['speech']
            report["detailed_analysis"].update({
                "filler_word_count": speech['filler_total'],
                "filler_ratio": round(speech['filler_ratio'], 1),
                "connective_count": speech['connective_total'],
                "vocabulary_richness": round(speech['vocabulary_richness'], 1),
                "avg_sentence_length": speech['avg_sentence_length']
            })
            
            report["ai_details"] = {
                "recognized_text": speech['full_text'][:500],
                "filler_words": speech['filler_words'],
                "connectives": speech['connectives'],
                "top_nouns": speech['top_nouns']
            }

        if ai_analysis.get('tremor'):
            tremor = ai_analysis['tremor']
            report["detailed_analysis"].update({
                "jitter_percent": round(tremor['jitter_percent'], 2),
                "average_pitch": round(tremor['average_pitch'], 1)
            })

        # 리포트 저장
        voice_analysis_data["final_report"] = report
        voice_analysis_data["analysis_complete"] = True
        
        print("=" * 70)
        print("✅ 리포트 생성 완료!")
        print(f"   🎯 종합 점수: {report['overall_score']}")
        print(f"   🕐 타임스탬프: {report['analysis_timestamp']}")
        print(f"   🤖 AI 분석: {report['ai_powered']}")
        print("=" * 70)

        return report
        
    except Exception as e:
        print("=" * 70)
        print(f"❌ 리포트 생성 오류: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        
        # 오류 시 기본 리포트
        error_report = {
            "overall_score": 0,
            "analysis_timestamp": time.time(),
            "detailed_analysis": {
                "voice_confidence": 0,
                "average_volume": 0,
                "volume_consistency": 0,
                "voice_stability": 0
            },
            "recommendations": [f"분석 중 오류가 발생했습니다: {str(e)}"],
            "error": str(e),
            "ai_powered": False
        }
        
        voice_analysis_data["final_report"] = error_report
        voice_analysis_data["analysis_complete"] = True
        
        return error_report


def run_comprehensive_ai_analysis():
    try:
        print("🤖 AI 종합 분석 시작...")
        
        audio_file = "final_interview_audio.webm"
        
        if not os.path.exists(audio_file):
            print(f"⚠️ 오디오 파일 없음")
            generate_voice_analysis_report()
            return
        
        if LIBROSA_AVAILABLE:
            tremor_analysis = analyze_voice_tremor_librosa(audio_file)
            if tremor_analysis:
                voice_analysis_data["ai_analysis"]["tremor"] = tremor_analysis
        
        if WHISPER_AVAILABLE:
            speech_analysis = analyze_speech_with_whisper_konlpy(audio_file)
            if speech_analysis:
                voice_analysis_data["ai_analysis"]["speech"] = speech_analysis
        
        generate_voice_analysis_report()
        print("✅ AI 종합 분석 완료")
        
    except Exception as e:
        print(f"❌ AI 종합 분석 오류: {e}")
        generate_voice_analysis_report()

# 면접 자동 진행
def auto_next_phase():
    global interview_session
    
    if not interview_session['active']:
        return
    
    current_phase = interview_session['phase']
    
    if current_phase == "thinking":
        interview_session['phase'] = "answering"
        interview_session['phase_start_time'] = time.time()
        
        interview_session['auto_timer'] = threading.Timer(
            interview_session['answer_time'], 
            auto_next_phase
        )
        interview_session['auto_timer'].start()
        
    elif current_phase == "answering":
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
        
        interview_session['auto_timer'] = threading.Timer(
            interview_session['thinking_time'], 
            auto_next_phase
        )
        interview_session['auto_timer'].start()
        
    else:
        interview_session['phase'] = 'finished'
        voice_analysis_data["session_active"] = False
        threading.Thread(target=run_comprehensive_ai_analysis, daemon=True).start()
        threading.Timer(3.0, stop_interview_internal).start()

def stop_interview_internal():
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

def generate_structured_interview_questions():
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
    
    return structured_questions

# ==================== WebRTC API 엔드포인트들 ====================

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'mode': 'WebRTC',
        'voice_analysis': 'AI-Powered' if (WHISPER_AVAILABLE and LIBROSA_AVAILABLE) else 'Basic',
        'ai_modules': {
            'whisper': WHISPER_AVAILABLE,
            'konlpy': KONLPY_AVAILABLE,
            'librosa': LIBROSA_AVAILABLE,
            'pydub': AUDIO_PROCESSING_AVAILABLE,
            'mediapipe': pose is not None
        }
    })

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """클라이언트에서 전송한 프레임을 분석"""
    global latest_analysis_data, is_analysis_active, calibration_in_progress
    
    try:
        data = request.get_json()
        
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data'}), 400
        
        # Base64 디코딩
        frame_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        frame_bytes = base64.b64decode(frame_data)
        
        # numpy 배열로 변환
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400
        
        # MediaPipe 자세 분석
        result = {
            'timestamp': time.time(),
            'calibration_in_progress': calibration_in_progress,
            'analysis_active': is_analysis_active
        }
        
        if pose:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                if calibration_in_progress:
                    analyzer.add_calibration_sample(pose_results.pose_landmarks.landmark)
                    result['calibration_status'] = 'collecting'
                    result['samples_collected'] = len(analyzer.calibration_samples)
                    
                elif analyzer.calibrated and is_analysis_active:
                    analysis = analyzer.analyze_posture(pose_results.pose_landmarks.landmark)
                    
                    # 면접 정보
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
                    
                    result.update({
                        'posture_score': analysis['score'],
                        'issues': analysis['issues'],
                        'in_frame': analysis['in_frame'],
                        'interview': interview_info
                    })
            else:
                result['error'] = 'No pose detected'
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ 프레임 분석 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    global is_analysis_active, calibration_in_progress, voice_analysis_data
    
    print("=" * 70)
    print("🎯 [START_ANALYSIS] 호출됨")
    print("=" * 70)
    
    # 음성 분석 데이터 완전 초기화
    print("🧹 음성 데이터 완전 초기화 중...")
    voice_analysis_data.clear()
    voice_analysis_data.update({
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
        "ai_analysis": {}
    })
    
    print(f"✅ 초기화 완료:")
    print(f"   - session_active: {voice_analysis_data['session_active']}")
    print(f"   - analysis_complete: {voice_analysis_data['analysis_complete']}")
    print(f"   - final_report: {bool(voice_analysis_data['final_report'])}")
    
    analyzer.reset_calibration()
    calibration_in_progress = True
    is_analysis_active = False
    
    print("=" * 70)
    return jsonify({'status': 'calibration_started'})

@app.route('/finalize_calibration', methods=['POST'])
def finalize_calibration():
    global calibration_in_progress, is_analysis_active
    
    success = analyzer.finalize_calibration()
    
    if success:
        calibration_in_progress = False
        is_analysis_active = True
        return jsonify({'status': 'calibration_complete'})
    else:
        return jsonify({'status': 'calibration_failed', 'message': '샘플 부족'}), 400

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    global is_analysis_active, calibration_in_progress, voice_analysis_data
    
    print("=" * 70)
    print("🛑 분석 중지 요청")
    print("=" * 70)
    
    is_analysis_active = False
    calibration_in_progress = False
    
    # 면접 중지
    if interview_session.get('auto_timer'):
        try:
            interview_session['auto_timer'].cancel()
            print("✅ 면접 타이머 취소")
        except:
            pass
    
    interview_session.update({
        'active': False, 
        'current_question': None,
        'phase': 'finished', 
        'auto_timer': None
    })
    
    # 음성 분석 데이터 완전 초기화
    print("🧹 음성 분석 데이터 초기화")
    voice_analysis_data.clear()
    voice_analysis_data.update({
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
        "ai_analysis": {}
    })
    
    print("=" * 70)
    print("✅ 분석 중지 완료")
    print("=" * 70)
    
    return jsonify({
        'status': 'stopped',
        'voice_data_reset': True
    })

@app.route('/interview/start', methods=['POST'])
def start_interview():
    global interview_session, voice_analysis_data
    
    print("=" * 70)
    print("🎬 [INTERVIEW/START] 호출됨")
    print("=" * 70)
    
    # 1. 이전 타이머 취소
    if interview_session.get('auto_timer'):
        try:
            interview_session['auto_timer'].cancel()
            print("✅ 이전 타이머 취소")
        except:
            pass
    
    # 2. 이전 오디오 파일 삭제
    print("🧹 오디오 파일 삭제 중...")
    try:
        if os.path.exists("final_interview_audio.webm"):
            os.remove("final_interview_audio.webm")
            print("   ✅ final_interview_audio.webm 삭제")
        
        deleted_chunks = 0
        for f in os.listdir('.'):
            if f.startswith('temp_chunk_') and f.endswith('.webm'):
                try:
                    os.remove(f)
                    deleted_chunks += 1
                except:
                    pass
        
        if deleted_chunks > 0:
            print(f"   ✅ {deleted_chunks}개 임시 청크 삭제")
            
    except Exception as e:
        print(f"   ⚠️ 파일 삭제 오류: {e}")
    
    # 3. voice_analysis_data 완전 초기화 (한 번만!)
    print("🔄 음성 데이터 완전 초기화 중...")
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
    
    print("✅ 초기화 완료:")
    print(f"   - session_active: {voice_analysis_data['session_active']}")
    print(f"   - analysis_complete: {voice_analysis_data['analysis_complete']}")
    print(f"   - final_report: {bool(voice_analysis_data['final_report'])}")
    
    # 4. 면접 질문 생성
    print("📝 질문 생성 중...")
    structured_questions = generate_structured_interview_questions()
    first_q = structured_questions[0]
    print(f"✅ {len(structured_questions)}개 질문 생성")
    
    # 5. 면접 세션 설정
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
    
    # 6. 자동 진행 타이머 시작
    interview_session['auto_timer'] = threading.Timer(
        interview_session['thinking_time'], 
        auto_next_phase
    )
    interview_session['auto_timer'].start()
    
    print("=" * 70)
    print(f"🎉 면접 시작 완료 - 총 {len(structured_questions)}개 질문")
    print("=" * 70)
    
    return jsonify({
        'status': 'started',
        'total_questions': len(structured_questions),
        'current_question': first_q['question'],
        'category': first_q['category'],
        'flow_stage': INTERVIEW_FLOW[first_q['stage']],
        'voice_analysis_active': True,
        'ai_enabled': WHISPER_AVAILABLE and LIBROSA_AVAILABLE,
        'voice_data_reset': True
    })

@app.route('/interview/stop', methods=['POST'])
def stop_interview():
    stop_interview_internal()
    return jsonify({'status': 'stopped'})

@app.route('/interview/status', methods=['GET'])
def interview_status():
    current_time = time.time()
    time_remaining = 0
    
    if interview_session['phase_start_time']:
        elapsed = int(current_time - interview_session['phase_start_time'])
        if interview_session['phase'] == 'thinking':
            time_remaining = max(0, interview_session['thinking_time'] - elapsed)
        elif interview_session['phase'] == 'answering':
            time_remaining = max(0, interview_session['answer_time'] - elapsed)
    
    return jsonify({
        'active': interview_session['active'],
        'current_question': interview_session['current_question'],
        'category': interview_session['category'],
        'question_number': interview_session['question_index'],
        'total_questions': len(interview_session['questions_list']),
        'phase': interview_session['phase'],
        'time_remaining': time_remaining,
        'current_stage': INTERVIEW_FLOW[interview_session.get('current_flow_stage', 0)] if interview_session['active'] else None
    })

# 음성 분석 엔드포인트 (기존과 동일)
@app.route('/voice/audio_chunk_blob', methods=['POST'])
def receive_audio_chunk_blob():
    try:
        if not voice_analysis_data["session_active"]:
            return jsonify({'status': 'session_inactive'})
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'})
        
        audio_file = request.files['audio']
        chunk_number = int(request.form.get('chunk_number', 0))
        audio_data = audio_file.read()
        voice_analysis_data["chunk_count"] += 1
        
        chunk_path = f"temp_chunk_{chunk_number}.webm"
        with open(chunk_path, 'wb') as f:
            f.write(audio_data)
        
        if len(audio_data) > 100:
            basic_volume = min(100, (len(audio_data) / 1000) * 8)
            basic_sample = {
                'volume': basic_volume / 100,
                'is_speaking': True,
                'timestamp': time.time()
            }
            voice_analysis_data["audio_samples"].append(basic_sample)
        
        return jsonify({
            'status': 'success',
            'chunk_number': chunk_number,
            'samples_total': len(voice_analysis_data["audio_samples"])
        })
        
    except Exception as e:
        print(f"❌ 청크 처리 실패: {e}")
        return jsonify({'error': str(e)})

@app.route('/voice/final_audio', methods=['POST'])
def receive_final_audio():
    try:
        if 'final_audio' not in request.files:
            return jsonify({'error': 'No final audio file'})
        
        audio_file = request.files['final_audio']
        final_audio_path = "final_interview_audio.webm"
        audio_file.save(final_audio_path)
        
        file_size = os.path.getsize(final_audio_path)
        
        if WHISPER_AVAILABLE or LIBROSA_AVAILABLE:
            threading.Thread(target=run_comprehensive_ai_analysis, daemon=True).start()
        
        return jsonify({
            'status': 'final_audio_received',
            'file_size': file_size,
            'ai_analysis_started': WHISPER_AVAILABLE or LIBROSA_AVAILABLE
        })
        
    except Exception as e:
        print(f"❌ 최종 오디오 파일 처리 실패: {e}")
        return jsonify({'error': str(e)})

@app.route('/voice/analysis_report', methods=['GET'])
def get_voice_analysis_report():
    """음성 분석 리포트 조회"""
    
    print("=" * 50)
    print("📥 [ANALYSIS_REPORT] 요청 수신")
    print(f"   - session_active: {voice_analysis_data.get('session_active')}")
    print(f"   - analysis_complete: {voice_analysis_data.get('analysis_complete')}")
    print(f"   - final_report 존재: {bool(voice_analysis_data.get('final_report'))}")
    
    # ===== 핵심: 세션 활성 중에는 리포트 반환 안 함 =====
    if voice_analysis_data.get("session_active"):
        print("   ⚠️ 세션 활성 중 - 리포트 없음")
        print("=" * 50)
        return jsonify({
            'status': 'session_active',
            'message': '면접이 진행 중입니다.',
            'analysis_complete': False
        })
    
    # analysis_complete 체크
    if voice_analysis_data.get("analysis_complete") and voice_analysis_data.get("final_report"):
        report = voice_analysis_data["final_report"]
        
        # 타임스탬프 확인
        if 'analysis_timestamp' not in report:
            report['analysis_timestamp'] = time.time()
            print("   ⚠️ 타임스탬프 없음 - 생성")
        
        print(f"   ✅ 리포트 전송:")
        print(f"      - 점수: {report.get('overall_score')}")
        print(f"      - 타임스탬프: {report.get('analysis_timestamp')}")
        print(f"      - AI 분석: {report.get('ai_powered')}")
        print("=" * 50)
        
        return jsonify(report)
    else:
        # 아직 분석 중
        samples = len(voice_analysis_data.get("audio_samples", []))
        chunks = voice_analysis_data.get("chunk_count", 0)
        
        print(f"   ⏳ 리포트 준비 중:")
        print(f"      - 샘플: {samples}개")
        print(f"      - 청크: {chunks}개")
        print("=" * 50)
        
        return jsonify({
            'status': 'analysis_in_progress',
            'message': 'AI 음성 분석이 진행 중입니다. 잠시만 기다려주세요.',
            'ai_enabled': WHISPER_AVAILABLE and LIBROSA_AVAILABLE,
            'samples': samples,
            'chunks': chunks,
            'analysis_complete': False
        })

# 리소스 정리
def cleanup_resources():
    print("🧹 리소스 정리 시작...")
    
    global pose
    if pose:
        try:
            pose.close()
            print("✅ MediaPipe 리소스 해제")
        except:
            pass
        pose = None
    
    if interview_session.get('auto_timer'):
        try:
            interview_session['auto_timer'].cancel()
        except:
            pass
    
    try:
        for f in os.listdir('.'):
            if f.startswith('temp_chunk_') or f == 'final_interview_audio.webm':
                os.remove(f)
    except:
        pass
    
    print("🧹 리소스 정리 완료")

atexit.register(cleanup_resources)

if __name__ == '__main__':
    print("=" * 90)
    print("🎓 AI 기반 체육대학 면접 분석 시스템 (WebRTC 모드)")
    print("=" * 90)
    print(f"🌐 서버 주소: http://0.0.0.0:5001")
    print("📱 모드: WebRTC (클라이언트 브라우저 카메라 사용)")
    print("\n🤖 AI 모듈 상태:")
    print(f"   - Whisper: {'✅' if WHISPER_AVAILABLE else '❌'}")
    print(f"   - KoNLPy: {'✅' if KONLPY_AVAILABLE else '❌'}")
    print(f"   - librosa: {'✅' if LIBROSA_AVAILABLE else '❌'}")
    print(f"   - MediaPipe: {'✅' if pose else '❌'}")
    print("=" * 90)

    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 중...")
        cleanup_resources()
    except Exception as e:
        print(f"\n💥 오류: {e}")
        cleanup_resources()