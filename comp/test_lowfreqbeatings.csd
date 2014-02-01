<CsoundSynthesizer>
<CsOptions>
-odac
</CsOptions>
<CsInstruments>
;example by Joachim Heintz
sr = 44100
ksmps = 128
nchnls = 2
0dbfs = 1

instr 1
iamp_sin = ampdb(-6)
iamp_instr = ampdb(-12)
idur = 4
iwet = 0.1
ilfofreq = 4/3
ipch1 = 6.07
ipch1b = 6.04
ipch2 = 6.07
i_deltafreq2 = -4

k1_freq = linseg(cpspch(ipch1), idur*0.8, cpspch(ipch1b), idur*0.2, cpspch(ipch1))
;a1    oscili  iamp_instr, k1_freq
k1_spectrum = linseg(0.3, idur, 0.6)
a1 gbuzz iamp_instr, k1_freq, 12, 1, k1_spectrum, 1
; a1 += oscili(iamp_instr, cpspch(ipch1))
a1_env = linseg(0, idur*0.05, 1, idur*0.9, 1, idur*0.05, 0)
a1 *= a1_env

k1_pos = linseg(1, idur*0.9, 0, idur*0.1, 1)
;k1_pos = 0.5
a1L, a1R pan2 a1, k1_pos

k2_freq = linseg(cpspch(ipch2)+i_deltafreq2, idur, cpspch(ipch2)+i_deltafreq2)
a2	oscili iamp_instr, k2_freq
a2 *= a1_env

;a2 vco2 iamp_instr, k2_freq, 0
;a2 buzz iamp_instr, k2_freq, 10, 1
; a2 *= oscili(0.5, ilfofreq)+0.5 
a2L, a2R pan2 a2, k1_pos

k3_freq = linseg(cpspch(ipch1), idur, cpspch(ipch2))

;a3 oscili iamp_sin, k3_freq
a3 = 0

a3_pan = oscili(0.5, ilfofreq*0.5) + 0.5

a3L, a3R pan2 a3, a3_pan


aL = a1L+a2L+a3L
aR = a1R+a2R+a3R
aLw, aRw reverbsc aL, aR, 0.8, 10000, 44100, 0.2
aL  ntrpol aL, aLw, iwet
aR  ntrpol aR, aRw, iwet
      outs    aL,aR
endin

</CsInstruments>
<CsScore>
f 1 0 16384 10 1
i 1 0 99999;1000 Hz tone
</CsScore>
</CsoundSynthesizer>
<bsbPanel>
 <label>Widgets</label>
 <objectName/>
 <x>100</x>
 <y>100</y>
 <width>320</width>
 <height>240</height>
 <visible>true</visible>
 <uuid/>
 <bgcolor mode="nobackground">
  <r>255</r>
  <g>255</g>
  <b>255</b>
 </bgcolor>
</bsbPanel>
<bsbPresets>
</bsbPresets>
