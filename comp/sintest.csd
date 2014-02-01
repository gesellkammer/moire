<CsoundSynthesizer>
<CsOptions>
-o dac
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
nchnls = 2; change here if your output device has more channels
0dbfs = 1


giSine     ftgen      0,0, 2^10, 10, 1 

instr 1
	kamp_instr init 0.6
	kamp_sin init 0.6
	a1 = oscili(kamp_instr, 1036, giSine)
	a1L, a1R pan2 a1, 0
	a2 = oscili(kamp_instr, 1036, giSine)
	a2L, a2R pan2 a2, 1
	a3 = oscili(kamp_sin, 1024) * lfo(1, 1)
	a3L, a3R pan2 a3, 0.5
	aL = a1L+a2L+a3L
	aR = a1R+a2R+a3R
	;outs aL, aR
	outs a1, a1
endin


instr 2
asig       oscili     1,1000, giSine 
outch      1,asig 
endin

</CsInstruments>
<CsScore>
i 2 0 9999
f 1 0 4096 10 0 1
e
</CsScore>
<bsbPanel>
 <label>Widgets</label>
 <objectName/>
 <x>734</x>
 <y>681</y>
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
