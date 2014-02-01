FaustDelaysampl : UGen
{
  *ar { | in1, delay_samps(0.0) |
      ^this.multiNew('audio', in1, delay_samps)
  }

  *kr { | in1, delay_samps(0.0) |
      ^this.multiNew('control', in1, delay_samps)
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

  name { ^"FaustDelaysampl" }
}

