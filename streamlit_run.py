import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import whisper
from deep_translator import GoogleTranslator
import os
from pydub import AudioSegment
import asyncio
import edge_tts
from pydub.effects import speedup

st.set_page_config(page_title="Video Translation", layout="centered")
st.title("🎬 Automatic Video Translation")

voice_options = {
    "أنثى مصرية": {"ar": "ar-EG-SalmaNeural", "en": "en-US-JennyNeural"},
    "ذكر جزائري": {"ar": "ar-DZ-IsmaelNeural", "en": "en-US-EricNeural"},
    "أنثى سعودية": {"ar": "ar-SA-ZariyahNeural", "en": "en-US-EmmaMultilingualNeural"},
    "ذكر سعودي": {"ar": "ar-SA-HamedNeural", "en": "en-US-EricNeural"},
    "أنثى لبنانية": {"ar": "ar-LB-LaylaNeural", "en": "en-US-EmmaMultilingualNeural"},
}

voice_gender = st.selectbox("Select the type of voice", list(voice_options.keys()), index=0)
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

async def convert_text_to_speech(text, lang, voice_gender):
    voice = voice_options[voice_gender][lang]
    output_path = os.path.join(tempfile.gettempdir(), "voice.mp3")
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)
    return output_path

if uploaded_video:
    st.video(uploaded_video)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    tts_path = None 

    try:
        video_clip = VideoFileClip(video_path)
        audio_path = video_path.replace(".mp4", ".mp3")
        video_clip.audio.write_audiofile(audio_path)

        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        original_text = result["text"]

        source_lang = "ar" if result["language"] == "ar" else "en"
        target_lang = "en" if source_lang == "ar" else "ar"

        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(original_text)

        tts_path = asyncio.run(convert_text_to_speech(translated_text, target_lang, voice_gender))

        audio = AudioSegment.from_file(tts_path)
        original_audio_duration = audio.duration_seconds
        video_duration = video_clip.duration

        if not video_duration or video_duration == 0:
            st.error("❌ مدة الفيديو غير صالحة.")
            st.stop()

        speed_factor = original_audio_duration / video_duration

        if speed_factor == 0 or speed_factor == float('inf'):
            st.warning("⚠️ حصلت مشكلة في ضبط السرعة. سيتم استخدام الصوت كما هو.")
            new_audio = audio
        else:
            try:
                new_audio = speedup(audio, playback_speed=speed_factor, crossfade=0)
            except ZeroDivisionError:
                st.warning("⚠️ الصوت قصير جدًا للتعديل، سيتم استخدامه كما هو.")
                new_audio = audio

        adjusted_audio_path = os.path.join(tempfile.gettempdir(), "adjusted_voice.wav")
        new_audio.export(adjusted_audio_path, format="wav")

        final_path = os.path.join(tempfile.gettempdir(), "final_video.mp4")
        new_audio_clip = AudioFileClip(adjusted_audio_path)
        new_video = video_clip.set_audio(new_audio_clip)
        new_video.write_videofile(final_path, codec="libx264", audio_codec="aac")

        st.success("✅ تم الانتهاء! الفيديو بعد الترجمة:")
        st.video(final_path)

    finally:
        try:
            video_clip.reader.close()
            if video_clip.audio:
                video_clip.audio.reader.close_proc()
        except:
            pass
        for f in [video_path, audio_path, tts_path]:  # ← خليها زي كده
            if f and os.path.exists(f):    
                try:
                    os.remove(f)
                except:
                    pass
