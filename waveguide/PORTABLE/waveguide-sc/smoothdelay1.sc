FaustSmoothDelay1 : UGen
{
  *ar { | in1, delay(0.0), feedback(0.0), interpolation(10.0) |
      /*
        delay: ms (0-5000)
        interpolation: ms
        feedback: 0-100
      */
      ^this.multiNew('audio', in1, delay, feedback, interpolation)
  }

  *kr { | in1, delay(0.0), feedback(0.0), interpolation(10.0) |
      ^this.multiNew('control', in1, delay, feedback, interpolation)
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

  name { ^"FaustSmoothDelay1" }
}

