import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Mic, BrainCircuit, XCircle, Activity, Eye, Clock } from 'lucide-react';

export const Cockpit = ({ sessionId, initialQuestion, config, onEnd }) => {
  const videoRef = useRef(null);
  const socketRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const faceMeshRef = useRef(null);
  const animationFrameRef = useRef(null);

  const [metrics, setMetrics] = useState({ 
    eye: 0, 
    fidget: 0, 
    smile: false, 
    stress: false,
    posture: {
      shoulder_angle: 0,
      is_leaning: false,
      is_slouching: false,
      slouch_score: 0,
      arms_crossed: false,
      shoulder_stability: 1.0
    }
  });
  const [transcript, setTranscript] = useState(initialQuestion || "System Online");
  const [aiState, setAiState] = useState('idle');
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [composure, setComposure] = useState(50);

  // Calculate composure score
  useEffect(() => {
    const eyeScore = metrics.eye * 50;
    const smileBonus = metrics.smile ? 25 : 0;
    const fidgetPenalty = Math.min(metrics.fidget * 2, 25);
    const stressPenalty = metrics.stress ? 10 : 0;
    
    const score = Math.round(eyeScore + smileBonus + (25 - fidgetPenalty) - stressPenalty);
    setComposure(Math.min(100, Math.max(0, score)));
  }, [metrics]);

  // Timer
  useEffect(() => {
    const timer = setInterval(() => setDuration(d => d + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const playAudio = (base64Audio) => {
    try {
      if (!base64Audio) return;
      const audio = new Audio(`data:audio/mp3;base64,${base64Audio}`);
      
      audio.onended = () => setAiState('idle');
      audio.onerror = (e) => {
        console.error("Audio playback error:", e);
        fallbackToTTS(transcript);
      };
      
      audio.play().catch(err => {
        console.error("Play failed:", err);
        fallbackToTTS(transcript);
      });
    } catch (e) {
      fallbackToTTS(transcript);
    }
  };

  const fallbackToTTS = (text) => {
    setAiState('speaking');
    const utt = new SpeechSynthesisUtterance(text);
    utt.onend = () => setAiState('idle');
    window.speechSynthesis.speak(utt);
  };

  // ----------------------------------------------------------------------
  // CORE LOGIC: Camera & Socket (Race Condition Fixed)
  // ----------------------------------------------------------------------
  useEffect(() => {
    let ws = null;
    let myStream = null;
    let isMounted = true; // <--- The Fix: Track mount status

    // 1. WebSocket Setup
    console.log("Connecting to WebSocket...");
    ws = new WebSocket(`ws://localhost:8000/ws/interview/${sessionId}`);
    socketRef.current = ws;

    ws.onopen = () => console.log("✅ WebSocket connected!");
    
    ws.onmessage = (event) => {
      if (!isMounted) return;
      const data = JSON.parse(event.data);
      
      if (data.type === 'metrics_update') {
        setMetrics({
          eye: data.metrics.eye_contact_score || 0,
          fidget: data.metrics.fidget_score || 0,
          smile: data.metrics.is_smiling || false,
          stress: data.metrics.is_stressed || false,
          posture: data.metrics.posture || {
            shoulder_angle: 0, is_leaning: false, is_slouching: false,
            slouch_score: 0, arms_crossed: false, shoulder_stability: 1.0
          }
        });
      }
      
      if (data.type === 'ai_response') {
        setAiState('speaking');
        setTranscript(data.reply);
        if (data.audio) playAudio(data.audio);
        else fallbackToTTS(data.reply);
      }
    };

    // 2. FaceMesh Helper
    const initializeFaceMesh = (videoElement) => {
       const FaceMesh = window.FaceMesh;
       if (!FaceMesh) return;
       
       faceMeshRef.current = new FaceMesh({
         locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
       });
       
       faceMeshRef.current.setOptions({
         maxNumFaces: 1,
         refineLandmarks: true,
         minDetectionConfidence: 0.5,
         minTrackingConfidence: 0.5
       });

       faceMeshRef.current.onResults((results) => {
          if (!isMounted) return;
          if (results.multiFaceLandmarks?.[0] && socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify({
              type: "tracking",
              landmarks: results.multiFaceLandmarks[0]
            }));
          }
       });

       const processFrame = async () => {
         if (!isMounted) return;
         if (videoElement && videoElement.readyState === 4) {
           await faceMeshRef.current.send({ image: videoElement });
         }
         animationFrameRef.current = requestAnimationFrame(processFrame);
       };
       processFrame();
    };

    // 3. Audio Setup Helper
    const setupAudioRecorder = (stream) => {
      try {
        const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', '']
          .find(m => m === '' || MediaRecorder.isTypeSupported(m));
        const recorder = mimeType
          ? new MediaRecorder(stream, { mimeType })
          : new MediaRecorder(stream);
        mediaRecorderRef.current = recorder;
        
        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) audioChunksRef.current.push(e.data);
        };
        
        recorder.onstop = () => {
          if (!isMounted) return;
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          const reader = new FileReader();
          reader.readAsDataURL(audioBlob);
          reader.onloadend = () => {
             const base64Audio = reader.result.split(',')[1];
             if (socketRef.current?.readyState === WebSocket.OPEN) {
               setAiState('processing');
               socketRef.current.send(JSON.stringify({
                 type: "conversation",
                 audio_data: base64Audio,
                 landmarks: []
               }));
             }
          };
          audioChunksRef.current = [];
        };
      } catch (e) {
        console.error("MediaRecorder failed:", e);
      }
    };

    // 4. Camera Setup with Zombie Prevention
    const setupCamera = async () => {
      try {
        if (videoRef.current && videoRef.current.srcObject) {
          videoRef.current.srcObject.getTracks().forEach(t => t.stop());
        }

        // Get video and audio as SEPARATE streams to prevent audio track death
        const videoStream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
        });

        const audioStream = await navigator.mediaDevices.getUserMedia({ 
          audio: { echoCancellation: true, noiseSuppression: true }
        });

        if (!isMounted) {
          videoStream.getTracks().forEach(t => t.stop());
          audioStream.getTracks().forEach(t => t.stop());
          return;
        }

        myStream = videoStream;
        console.log("✅ Camera granted and active");

        if (videoRef.current) {
          videoRef.current.srcObject = videoStream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            initializeFaceMesh(videoRef.current);
          };
        }

        // Setup recorder with AUDIO-ONLY stream
        setupAudioRecorder(audioStream);

      } catch (err) {
        if (!isMounted) return;
        console.error("Camera setup failed:", err);
        if (err.name === 'NotAllowedError') alert("Permission denied. Reset permissions in browser bar.");
        else if (err.name !== 'AbortError') alert("Camera Error: " + err.message);
      }
    };

    setupCamera();
    
    // CLEANUP
    return () => {
      console.log("🛑 Cleanup triggered");
      isMounted = false; // Prevents async callbacks from running
      
      if (ws) {
        ws.close();
        socketRef.current = null;
      }
      
      // Stop the stream we tracked locally
      if (myStream) {
        myStream.getTracks().forEach(track => {
          track.stop();
          console.log("Stopped camera track");
        });
      }
      
      if (faceMeshRef.current) faceMeshRef.current.close();
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [sessionId]); // End useEffect

  const toggleRecording = () => {
    if (!mediaRecorderRef.current) return;
    const state = mediaRecorderRef.current.state;
    if (isRecording) {
      if (state === 'recording') mediaRecorderRef.current.stop();
      setIsRecording(false);
      window.speechSynthesis.cancel();
    } else {
      if (state === 'inactive') {
        audioChunksRef.current = [];
        mediaRecorderRef.current.start();
        setIsRecording(true);
        setAiState('listening');
        setTranscript("Listening...");
      }
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getBorderColor = () => {
    if (composure >= 70) return 'border-emerald-500 shadow-emerald-500/50';
    if (composure >= 40) return 'border-yellow-500 shadow-yellow-500/50';
    return 'border-red-500 shadow-red-500/50';
  };

  return (
    <div className="h-screen w-full bg-black flex overflow-hidden font-sans">
      {/* Main Video Area */}
      <div className="relative flex-1 bg-gradient-to-br from-gray-900 via-black to-blue-900 flex flex-col items-center justify-center p-8">
        <div className="absolute top-8 left-8 right-8 flex items-center justify-between z-10">
          <div className="flex items-center gap-6">
            <div className="bg-black/70 backdrop-blur-xl px-6 py-3 rounded-full border border-white/20 flex items-center gap-3">
              <Clock size={20} className="text-blue-400" />
              <span className="font-mono text-xl font-bold">{formatTime(duration)}</span>
            </div>
            <div className="bg-black/70 backdrop-blur-xl px-6 py-3 rounded-full border border-white/20">
              <span className="text-sm text-gray-300">{config.topic}</span>
            </div>
          </div>
          <button
            onClick={() => onEnd(sessionId)}
            className="bg-red-500/20 hover:bg-red-500/30 backdrop-blur-xl px-6 py-3 rounded-full border border-red-500/50 flex items-center gap-2 transition-all"
          >
            <XCircle size={20} className="text-red-400" />
            <span className="text-red-400 font-medium">End Interview</span>
          </button>
        </div>

        <div className={`relative w-full max-w-5xl aspect-video rounded-3xl overflow-hidden border-4 transition-all duration-500 shadow-2xl ${getBorderColor()}`}>
          <video 
            ref={videoRef} 
            muted 
            playsInline
            className="w-full h-full object-cover transform -scale-x-100" 
          />
          <div className="absolute bottom-0 left-0 right-0 p-8 bg-gradient-to-t from-black/80 to-transparent pointer-events-none">
            <motion.div
              key={transcript}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center"
            >
              <span className="inline-block bg-black/70 backdrop-blur-md text-white px-8 py-4 rounded-2xl text-lg font-medium shadow-2xl border border-white/10 max-w-4xl">
                {transcript}
              </span>
            </motion.div>
          </div>
          <div className="absolute top-6 right-6 bg-black/70 backdrop-blur-xl px-6 py-4 rounded-2xl border border-white/20">
            <div className="text-xs text-gray-400 mb-2">COMPOSURE</div>
            <div className="flex items-center gap-3">
              <span className="text-3xl font-bold text-white">{composure}</span>
              <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full ${composure >= 70 ? 'bg-emerald-500' : composure >= 40 ? 'bg-yellow-500' : 'bg-red-500'}`}
                  style={{ width: `${composure}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="absolute bottom-8 left-8 right-8 flex items-center justify-center gap-4 z-10">
          <div className={`bg-black/70 backdrop-blur-xl px-6 py-4 rounded-2xl border border-white/20 flex items-center gap-3 ${metrics.eye >= 0.6 ? 'border-emerald-500' : 'border-red-500'}`}>
            <Eye size={24} className={metrics.eye >= 0.6 ? 'text-emerald-400' : 'text-red-400'} />
            <div>
              <div className="text-xs text-gray-400">Eye Contact</div>
              <div className="text-lg font-bold text-white">{Math.round(metrics.eye * 100)}%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Sidebar */}
      <div className="w-96 bg-black border-l border-white/10 flex flex-col overflow-y-auto">
        {/* AI State */}
        <div className="flex flex-col items-center justify-center p-8 border-b border-white/10">
          <div className={`relative w-32 h-32 rounded-full border-4 flex items-center justify-center transition-all ${
            aiState === 'speaking' ? 'border-blue-500 shadow-blue-500/50' :
            aiState === 'listening' ? 'border-red-500 shadow-red-500/50' : 'border-gray-700'
          } shadow-lg`}>
            {aiState === 'listening' ? <Mic className="text-red-500" size={52} /> : 
             aiState === 'processing' ? <div className="w-12 h-12 border-4 border-yellow-500 border-t-transparent rounded-full animate-spin" /> :
             <BrainCircuit className="text-blue-400" size={52} />}
          </div>
          <div className="mt-4 text-center">
            <div className="text-xs font-mono text-gray-500 tracking-widest">{aiState.toUpperCase()} MODE</div>
          </div>
        </div>

        {/* Posture Panel */}
        <div className="p-6 border-b border-white/10">
          <div className="text-xs font-mono text-gray-400 tracking-widest mb-4">POSTURE ANALYSIS</div>
          <div className="space-y-3">
            {/* Shoulder Angle */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400">Shoulder Angle</span>
              <span className={`text-sm font-bold ${metrics.posture.is_leaning ? 'text-red-400' : 'text-emerald-400'}`}>
                {metrics.posture.shoulder_angle?.toFixed(1)}° {metrics.posture.is_leaning ? '⚠' : '✓'}
              </span>
            </div>
            {/* Slouch */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400">Slouch</span>
              <span className={`text-sm font-bold ${metrics.posture.is_slouching ? 'text-red-400' : 'text-emerald-400'}`}>
                {metrics.posture.is_slouching ? `⚠ ${(metrics.posture.slouch_score * 100).toFixed(0)}%` : '✓ Good'}
              </span>
            </div>
            {/* Arms */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400">Arms</span>
              <span className={`text-sm font-bold ${metrics.posture.arms_crossed ? 'text-red-400' : 'text-emerald-400'}`}>
                {metrics.posture.arms_crossed ? '⚠ Crossed' : '✓ Open'}
              </span>
            </div>
            {/* Stability */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400">Stability</span>
              <span className={`text-sm font-bold ${metrics.posture.shoulder_stability > 0.7 ? 'text-emerald-400' : 'text-yellow-400'}`}>
                {((metrics.posture.shoulder_stability || 1) * 100).toFixed(0)}%
              </span>
            </div>
            {/* Overall posture bar */}
            <div className="mt-2">
              <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className={`h-full rounded-full transition-all ${
                  !metrics.posture.is_leaning && !metrics.posture.is_slouching && !metrics.posture.arms_crossed
                    ? 'bg-emerald-500' : 'bg-red-500'
                }`} style={{ width: `${metrics.posture.shoulder_stability * 100}%` }} />
              </div>
            </div>
          </div>
        </div>

        {/* Record Button */}
        <div className="p-6 mt-auto">
          <button
            onClick={toggleRecording}
            disabled={aiState === 'processing' || aiState === 'speaking'}
            className={`w-full py-8 rounded-2xl font-bold text-lg tracking-wider transition-all shadow-2xl flex items-center justify-center gap-3 ${
              isRecording ? 'bg-red-500 text-white' : 'bg-blue-600 text-white'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {isRecording ? "STOP RECORDING" : "HOLD TO SPEAK"}
          </button>
        </div>
      </div>
    </div>
  );
};