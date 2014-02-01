FaustSDelay : UGen
{
  *ar { | in1, delay_s(0.0), feedback(0.0), interpolation_ms(10.0) |
      ^this.multiNew('audio', in1, delay_s, feedback, interpolation_ms)
  }

  *kr { | in1, delay_s(0.0), feedback(0.0), interpolation_ms(10.0) |
      ^this.multiNew('control', in1, delay_s, feedback, interpolation_ms)
  } 

  checkInputs {
    if (rate == 'audio', {
      1.do({|i|
        if (inputs.at(i).rate != 'audio', {
          ^(" input at index " + i + "(" + inputs.at(i) + 
            ") is not audio rate");
        });
      });
    });
    ^this.checkValidInputs
  }

  name { ^"FaustSDelay" }
}

