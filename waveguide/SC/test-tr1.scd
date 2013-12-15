(
{
    var mic_v7 = SoundIn(Saffire40.adat3);
    var mic_v6 = SoundIn(Saffire40.adat2);
    var mic_v2 = SoundIn(Saffire40.adat1);
    var out_adat1 = Out(10);
    var out_adat2 = Out(11);
    var out_head1 = Out(2);
    var out_head2 = Out(3);
    PinkNoise.ar    !> out_head1;
    SinOsc.ar(1000) * 0.4 !> out_head2;
    mic_v7 !> Out(0);
    mic_v6 !> Out(1);

}.play
)

Saffire40