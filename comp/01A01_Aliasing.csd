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
idur = 2
iwet = 0.2
ilfofreq = 3
ipch1 = 9.00
ipch2 = 8.11

k1_freq = linseg(cpspch(ipch1), idur, cpspch(ipch1))
a1    oscili  iamp_instr, k1_freq
;a1 buzz iamp_instr, 1036, 10, 1

a1L, a1R pan2 a1, 0.25

k2_freq = linseg(cpspch(ipch2), idur, cpspch(ipch2))
a2	oscili iamp_instr, k2_freq
;a2 vco2 iamp_instr, 1036, 0
;a2 buzz iamp_instr, 1036, 10, 1
a2L, a2R pan2 a2, 0.75

k3_freq = linseg(cpspch(ipch1), idur, cpspch(ipch2))
a3 oscili iamp_sin, k3_freq
;a3 *= lfo(1, 0.4)

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
 <x>118</x>
 <y>166</y>
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
