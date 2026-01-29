'use client'
import { useState, useEffect, useCallback, useRef } from 'react'

interface VitalData {
    patient_id: string
    timestamp: string
    risk_score: number
    risk_category: string
    vitals: {
        heart_rate: number
        systolic_bp: number
        diastolic_bp: number
        respiratory_rate: number
        spo2: number
        temperature: number
    }
    vital_trends: Record<string, number[]>
    risk_history: { timestamp: string; risk_score: number }[]
    patterns: { pattern_name: string; confidence: number; description: string }[]
    reasoning?: { severity: string; primary_concern: string; physiological_interpretation: string }
    alert?: { level: string; message: string }
    feature_importance: Record<string, number>
    patient_info?: { name: string; gestational_age?: string; birth_weight?: string; age_days?: number }
    ml_output?: {
        model_name: string
        task: string
        risk_score: number
        confidence: number
        prediction_time: string
    }
}

export default function Dashboard() {
    const [patients, setPatients] = useState<Map<string, VitalData>>(new Map())
    const [selectedPatient, setSelectedPatient] = useState<string | null>(null)
    const wsRef = useRef<WebSocket | null>(null)

    useEffect(() => {
        let ws: WebSocket | null = null;
        let reconnectTimeout: NodeJS.Timeout;

        const connect = () => {
            ws = new WebSocket('ws://localhost:8000/ws/vitals');
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('Connected to TriSense AI');
                ws?.send(JSON.stringify({ type: 'subscribe' }));
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.patient_id) {
                        setPatients(prev => new Map(prev).set(data.patient_id, data));
                        if (!selectedPatient) setSelectedPatient(data.patient_id);
                    }
                } catch (e) {
                    console.error("Error parsing WS message:", e);
                }
            };

            ws.onerror = (e) => console.error('WebSocket error:', e);

            ws.onclose = () => {
                console.log('Disconnected. Reconnecting in 3s...');
                reconnectTimeout = setTimeout(connect, 3000);
            };
        };

        connect();

        return () => {
            if (ws) ws.close();
            clearTimeout(reconnectTimeout);
        };
    }, []);

    const currentPatient = selectedPatient ? patients.get(selectedPatient) : null

    const getRiskColor = (category: string) => {
        const colors: Record<string, string> = {
            LOW: '#22c55e', MODERATE: '#eab308', HIGH: '#f97316', CRITICAL: '#ef4444'
        }
        return colors[category] || '#94a3b8'
    }

    const formatTime = (timestamp: string) => {
        return new Date(timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
    }

    const formatDate = (timestamp: string) => {
        return new Date(timestamp).toLocaleString('en-US', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })
    }

    return (
        <div className="min-h-screen bg-gray-100 font-sans">
            {/* Header */}
            <header className="bg-white border-b px-6 py-4 flex items-center justify-between shadow-sm">
                <div className="flex items-center gap-4">
                    <span className="font-extrabold text-2xl tracking-tight text-blue-600">TRISENSE AI</span>
                    <span className="text-[10px] bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full font-bold border border-blue-100">PROD v1.2</span>
                </div>
                <div className="flex items-center gap-6 text-sm font-medium text-gray-500">
                    <span className="bg-gray-50 px-3 py-1 rounded-lg border border-gray-100">🕒 {new Date().toLocaleDateString()}</span>
                    <button className="hover:text-red-500 transition-colors">Logout 🚪</button>
                </div>
            </header>

            <div className="flex h-[calc(100vh-73px)]">
                {/* Sidebar - Patient Info */}
                <aside className="w-80 bg-white border-r p-6 overflow-auto custom-scrollbar">
                    {currentPatient?.patient_info && (
                        <>
                            <div className="mb-8">
                                <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-1">Active Patient</div>
                                <h2 className="text-2xl font-black text-gray-900 leading-tight">{currentPatient.patient_info.name}</h2>
                                <div className="text-[10px] text-gray-400 mt-1">ID: {currentPatient.patient_id}</div>
                            </div>

                            <div className="mb-8 bg-gray-50 p-4 rounded-xl border border-gray-100">
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">Clinical Data</h3>
                                <div className="space-y-2.5">
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm text-gray-500">Postnatal Age</span>
                                        <span className="text-sm font-bold text-gray-700">{currentPatient.patient_info.age_days} Days</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm text-gray-500">Gestational Age</span>
                                        <span className="text-sm font-bold text-gray-700">{currentPatient.patient_info.gestational_age}</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm text-gray-500">Birth Weight</span>
                                        <span className="text-sm font-bold text-gray-700">{currentPatient.patient_info.birth_weight}</span>
                                    </div>
                                </div>
                            </div>

                            <div className="mb-6">
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">System Alerts</h3>
                                {currentPatient.alert ? (
                                    <div className={`p-4 rounded-xl text-sm shadow-sm ${currentPatient.alert.level === 'CRITICAL' ? 'bg-red-50 border border-red-100 text-red-700' : 'bg-orange-50 border border-orange-100 text-orange-700'}`}>
                                        <div className="font-bold mb-1 flex items-center gap-2">
                                            <span className="w-2 h-2 rounded-full bg-current animate-pulse" />
                                            {currentPatient.alert.level}
                                        </div>
                                        <div className="text-xs opacity-80 leading-relaxed">{currentPatient.alert.message?.slice(0, 120)}...</div>
                                    </div>
                                ) : (
                                    <div className="text-xs text-gray-400 p-4 bg-gray-50 rounded-xl border border-dashed border-gray-200 text-center italic">
                                        No active clinical alerts
                                    </div>
                                )}
                            </div>
                        </>
                    )}

                    {/* Patient List */}
                    <div className="mt-8">
                        <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-4">Patient Monitoring List</h3>
                        <div className="space-y-2">
                            {Array.from(patients.values()).map(p => (
                                <div
                                    key={p.patient_id}
                                    onClick={() => setSelectedPatient(p.patient_id)}
                                    className={`p-3 rounded-xl cursor-pointer transition-all border ${selectedPatient === p.patient_id ? 'bg-blue-50 border-blue-200 shadow-sm' : 'hover:bg-gray-50 border-transparent'}`}
                                >
                                    <div className="flex items-center gap-3">
                                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: getRiskColor(p.risk_category) }} />
                                        <span className="text-sm font-semibold text-gray-700">{p.patient_info?.name || p.patient_id}</span>
                                        <span className="text-[10px] font-bold text-gray-400 ml-auto">{(p.risk_score * 100).toFixed(0)}%</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </aside>

                {/* Main Content */}
                <main className="flex-1 p-8 overflow-auto bg-slate-50">
                    {/* ALERT BANNER - Shows when risk exceeds moderate threshold */}
                    {currentPatient && currentPatient.risk_score >= 0.35 && (
                        <div className={`mb-6 p-4 rounded-2xl border-2 shadow-lg animate-pulse ${currentPatient.risk_category === 'CRITICAL'
                            ? 'bg-red-50 border-red-300'
                            : currentPatient.risk_category === 'HIGH'
                                ? 'bg-orange-50 border-orange-300'
                                : 'bg-yellow-50 border-yellow-300'
                            }`}>
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    <div className={`w-12 h-12 rounded-full flex items-center justify-center ${currentPatient.risk_category === 'CRITICAL'
                                        ? 'bg-red-500'
                                        : currentPatient.risk_category === 'HIGH'
                                            ? 'bg-orange-500'
                                            : 'bg-yellow-500'
                                        }`}>
                                        <span className="text-2xl">⚠️</span>
                                    </div>
                                    <div>
                                        <div className={`text-lg font-black uppercase tracking-wide ${currentPatient.risk_category === 'CRITICAL'
                                            ? 'text-red-700'
                                            : currentPatient.risk_category === 'HIGH'
                                                ? 'text-orange-700'
                                                : 'text-yellow-700'
                                            }`}>
                                            {currentPatient.risk_category === 'CRITICAL'
                                                ? '🚨 CRITICAL SEPSIS RISK ALERT'
                                                : currentPatient.risk_category === 'HIGH'
                                                    ? '⚠️ HIGH SEPSIS RISK DETECTED'
                                                    : '⚡ MODERATE SEPSIS RISK'}
                                        </div>
                                        <div className={`text-sm font-medium mt-1 ${currentPatient.risk_category === 'CRITICAL'
                                            ? 'text-red-600'
                                            : currentPatient.risk_category === 'HIGH'
                                                ? 'text-orange-600'
                                                : 'text-yellow-600'
                                            }`}>
                                            {currentPatient.alert?.message
                                                ? currentPatient.alert.message.slice(0, 150) + '...'
                                                : `Risk Score: ${(currentPatient.risk_score * 100).toFixed(1)}% — ${currentPatient.reasoning?.primary_concern || 'Requires clinical attention'}`}
                                        </div>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className={`text-3xl font-black tabular-nums ${currentPatient.risk_category === 'CRITICAL'
                                        ? 'text-red-600'
                                        : currentPatient.risk_category === 'HIGH'
                                            ? 'text-orange-600'
                                            : 'text-yellow-600'
                                        }`}>
                                        {(currentPatient.risk_score * 100).toFixed(0)}%
                                    </div>
                                    <div className="text-xs font-bold text-gray-400 uppercase">
                                        {formatTime(currentPatient.timestamp)}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                    <div className="max-w-6xl mx-auto">
                        <div className="flex justify-between items-end mb-8">
                            <div>
                                <h1 className="text-4xl font-black text-gray-900 tracking-tight">Sepsis Risk Assessment</h1>
                                <p className="text-gray-500 mt-2 font-medium flex items-center gap-2">
                                    <span className="w-2 h-2 rounded-full bg-green-500" />
                                    Continuous Monitoring Active
                                </p>
                            </div>
                            <div className="text-right">
                                <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Last Updated</div>
                                <div className="text-sm font-bold text-gray-700">{currentPatient ? formatDate(currentPatient.timestamp) : '--'}</div>
                            </div>
                        </div>

                        <div className="grid grid-cols-12 gap-6">
                            {/* Left Panel: Risk & AI */}
                            <div className="col-span-12 lg:col-span-7 space-y-6">
                                {/* ML Regression Score Board */}
                                <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
                                    <div className="flex justify-between items-start mb-6">
                                        <div>
                                            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-1">Model Output</h3>
                                            <div className="flex items-baseline gap-2">
                                                <span className="text-5xl font-black tabular-nums" style={{ color: getRiskColor(currentPatient?.risk_category || 'LOW') }}>
                                                    {currentPatient?.ml_output ? currentPatient.ml_output.risk_score.toFixed(3) : '0.000'}
                                                </span>
                                                <span className="text-gray-400 font-bold text-lg">/ 1.0</span>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className="bg-gray-50 px-3 py-1 rounded-full border border-gray-100 inline-block mb-1">
                                                <span className="text-[10px] font-bold text-gray-500 uppercase">Model:</span>
                                                <span className="text-[10px] font-black text-blue-600 ml-1">{currentPatient?.ml_output?.model_name || 'PatchTST'}</span>
                                            </div>
                                            <div className="text-[10px] font-bold text-gray-400">Task: {currentPatient?.ml_output?.task || 'Sepsis Regression'}</div>
                                        </div>
                                    </div>

                                    <div className="bg-slate-50 rounded-xl p-4 flex justify-between items-center mb-6">
                                        <div className="flex items-center gap-4">
                                            <div className="text-center px-4 border-r border-gray-200">
                                                <div className="text-[10px] font-bold text-gray-400 uppercase mb-1">Confidence</div>
                                                <div className="text-xl font-black text-gray-800">{(currentPatient?.ml_output?.confidence || 0.91) * 100}%</div>
                                            </div>
                                            <div className="text-center px-4">
                                                <div className="text-[10px] font-bold text-gray-400 uppercase mb-1">ML Prediction Time</div>
                                                <div className="text-sm font-bold text-gray-600">{currentPatient?.ml_output ? formatDate(currentPatient.ml_output.prediction_time) : '--'}</div>
                                            </div>
                                        </div>
                                        <div className="flex gap-1">
                                            {[1, 2, 3, 4, 5].map(i => (
                                                <div key={i} className={`w-1.5 h-6 rounded-full ${i <= (currentPatient?.ml_output?.confidence || 0.9) * 5 ? 'bg-blue-500' : 'bg-gray-200'}`} />
                                            ))}
                                        </div>
                                    </div>

                                    {/* Score Trend Mini Graph */}
                                    <div className="h-20 w-full bg-gray-50 rounded-lg overflow-hidden relative">
                                        {currentPatient?.risk_history && currentPatient.risk_history.length > 0 && (
                                            <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                                                <defs>
                                                    <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.2" />
                                                        <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                                                    </linearGradient>
                                                </defs>
                                                <path
                                                    d={`${currentPatient.risk_history.slice(-20).map((point, i) =>
                                                        `${i === 0 ? 'M' : 'L'} ${(i / 19) * 100} ${100 - point.risk_score * 100}`
                                                    ).join(' ')} L 100 100 L 0 100 Z`}
                                                    fill="url(#scoreGrad)"
                                                />
                                                <path
                                                    d={currentPatient.risk_history.slice(-20).map((point, i) =>
                                                        `${i === 0 ? 'M' : 'L'} ${(i / 19) * 100} ${100 - point.risk_score * 100}`
                                                    ).join(' ')}
                                                    fill="none"
                                                    stroke="#3b82f6"
                                                    strokeWidth="2"
                                                    strokeLinecap="round"
                                                />
                                            </svg>
                                        )}
                                        <div className="absolute inset-0 flex items-center justify-center opacity-10 pointer-events-none">
                                            <span className="text-xs font-black uppercase tracking-[0.2em] text-blue-900">TEMPORAL RISK TREND</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Clinical Reasoning Card */}
                                {currentPatient?.reasoning && (
                                    <div className="bg-white rounded-2xl overflow-hidden shadow-sm border border-gray-100 flex flex-col">
                                        <div className="bg-slate-900 px-6 py-4 flex justify-between items-center">
                                            <span className="text-xs font-black text-white uppercase tracking-widest">STRICT AI EXPLAINER (NVIDIA QWEN)</span>
                                            <span className="text-[10px] bg-white/10 text-white/60 px-2 py-1 rounded font-mono">NON-DIAGNOSTIC LAYER</span>
                                        </div>
                                        <div className="p-6 space-y-4">
                                            <div className="bg-blue-50/50 p-4 rounded-xl border border-blue-100/50">
                                                <div className="text-[10px] font-bold text-blue-500 uppercase mb-1">Model Indicator</div>
                                                <div className="text-lg font-black text-blue-900 leading-tight">
                                                    {currentPatient.reasoning.severity}
                                                </div>
                                            </div>

                                            <div>
                                                <div className="text-[10px] font-bold text-gray-400 uppercase mb-2">Technical Explanation</div>
                                                <div className="text-sm text-gray-600 leading-relaxed font-medium bg-gray-50 p-5 rounded-2xl border border-gray-100 italic">
                                                    "{currentPatient.reasoning.physiological_interpretation}"
                                                </div>
                                            </div>

                                            <div className="flex items-center gap-2 pt-2 text-[10px] text-gray-400 font-bold border-t border-gray-100">
                                                <span className="w-1.5 h-1.5 rounded-full bg-orange-400" />
                                                This agent only explains model outputs. No clinical inference or diagnosis has been applied.
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Right Panel: Vitals */}
                            <div className="col-span-12 lg:col-span-5 space-y-6">
                                {/* Feature Importance */}
                                <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
                                    <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Regression Weights</h3>
                                    <div className="space-y-4">
                                        {currentPatient?.feature_importance && Object.entries(currentPatient.feature_importance).slice(0, 6).map(([key, val]) => (
                                            <div key={key}>
                                                <div className="flex justify-between text-[10px] font-bold mb-1 opacity-70 italic text-gray-500">
                                                    <span className="uppercase">{key.replace(/_/g, ' ')}</span>
                                                    <span>{(val * 100).toFixed(1)}%</span>
                                                </div>
                                                <div className="w-full bg-gray-100 h-1.5 rounded-full overflow-hidden">
                                                    <div className="h-full bg-slate-700 rounded-full" style={{ width: `${Math.min(val * 400, 100)}%` }} />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Vital Grids */}
                                <div className="grid grid-cols-2 gap-4">
                                    {[
                                        { label: 'HEART RATE', value: currentPatient?.vitals?.heart_rate, unit: 'BPM', color: 'text-rose-500', trend: currentPatient?.vital_trends?.heart_rate },
                                        { label: 'BLOOD OXYGEN', value: currentPatient?.vitals?.spo2, unit: '%', color: 'text-indigo-500', trend: currentPatient?.vital_trends?.spo2 },
                                        { label: 'RESPIRATION', value: currentPatient?.vitals?.respiratory_rate, unit: '/min', color: 'text-emerald-500', trend: currentPatient?.vital_trends?.respiratory_rate },
                                        { label: 'SYSTOLIC BP', value: currentPatient?.vitals?.systolic_bp, unit: 'mmHg', color: 'text-amber-500', trend: currentPatient?.vital_trends?.systolic_bp }
                                    ].map((v, idx) => (
                                        <div key={idx} className="bg-white rounded-2xl p-4 shadow-sm border border-gray-100">
                                            <div className="text-[10px] font-bold text-gray-400 uppercase mb-2 leading-none">{v.label}</div>
                                            <div className="flex items-baseline gap-1">
                                                <span className={`text-2xl font-black tabular-nums ${v.color}`}>{v.value?.toFixed(0) || '--'}</span>
                                                <span className="text-[9px] font-bold text-gray-300">{v.unit}</span>
                                            </div>
                                            {v.trend && (
                                                <div className="h-8 mt-2 opacity-50">
                                                    <svg className="w-full h-full" viewBox="0 0 100 50" preserveAspectRatio="none">
                                                        <path
                                                            d={v.trend.slice(-15).map((val, i) =>
                                                                `${i === 0 ? 'M' : 'L'} ${(i / 14) * 100} ${50 - (val / 200) * 50}`
                                                            ).join(' ')}
                                                            fill="none"
                                                            stroke="currentColor"
                                                            className={v.color}
                                                            strokeWidth="3"
                                                            strokeLinecap="round"
                                                        />
                                                    </svg>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Legend */}
                        <div className="flex justify-center gap-8 mt-12 py-6 border-t border-gray-200">
                            {[
                                { color: 'bg-green-500', label: 'Stable Operation' },
                                { color: 'bg-orange-500', label: 'Requires Observation' },
                                { color: 'bg-red-500', label: 'Critical Condition' }
                            ].map((l, i) => (
                                <div key={i} className="flex items-center gap-2">
                                    <div className={`w-2.5 h-2.5 rounded-full ${l.color}`} />
                                    <span className="text-[10px] font-extrabold text-gray-400 uppercase tracking-widest">{l.label}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </main>
            </div>
        </div>
    )
}
