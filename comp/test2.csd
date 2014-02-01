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
iamp_sin = 0.2
iamp_instr = 0.1

k1_freq linseg cpspch(p4), p3, cpspch(p5)
a1    oscili  iamp_instr, k1_freq
a1L, a1R pan2 a1, 0.5

a2	oscili iamp_instr, cpspch(p6)
;a2b oscili iamp_instr, cpspch(8.09)
;a2 = (a2+a2b)

a2L, a2R pan2 a2, 0.5

apos = oscili(0.5, p7)+0.5
kpos = downsamp(apos)

amixL = a1L * apos + a2L * (1-apos)
amixR = a1R * apos + a2R * (1-apos)
;amixL = ntrpol(a1L, a2L, kpos)
;amixR = ntrpol(a1R, a2R, kpos)

aenv = adsr(p3*0.2, 1, 1, p3*0.2)

outs amixL * aenv, amixR * aenv
endin

</CsInstruments>
<CsScore>
f 1 0 16384 10 1
i 1 0 5 8.07 8.08 8.09 0.5;1000 Hz tone
i 1 + 5 .    .    8.065 1;1000 Hz tone

</CsScore>
</CsoundSynthesizer>
cpspch
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
