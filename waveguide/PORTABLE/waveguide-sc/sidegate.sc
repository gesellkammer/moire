FaustSideGate : UGen
{
  *ar { | in1, in2, threshold(-30.0), attack(10.0), hold(200.0), release(100.0) |
      ^this.multiNew('audio', in1, in2, threshold, attack, hold, release)
  }

  *kr { | in1, in2, threshold(-30.0), attack(10.0), hold(200.0), release(100.0) |
      ^this.multiNew('control', in1, in2, threshold, attack, hold, release)
  } 

  checkInputs {
    if (rate == 'audio', {
      2.do({|i|
        if (inputs.at(i).rate != 'audio', {
          ^(" input at index " + i + "(" + inputs.at(i) + 
            ") is not audio rate");
        });
      });
    });
    ^this.checkValidInputs
  }

  name { ^"FaustSideGate" }
}

