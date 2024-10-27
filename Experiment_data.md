The experiment data on Painting 91 for optimal parameter finding.
1. test base_lr： 0.05, 0.03, 0.01, 0.005, 0.001

| base_lr |best_accuracy|last_accuracy|information|
|---------|-|-|-|
| 0.05    |0.670168|0.653361|lr is too large, the loss fluctuates
| 0.03    |0.697479|0.634454|the loss fluctuates
| 0.01    |0.710084|0.649160|no fluctuates, but loss reduce slowly, maybe test 0.012, 0.01, 0.008 with longer epoch will be better
| 0.005   |0.691176|0.661765|loss reduce slowly
| 0.001   |0.695378|0.682773|loss reduce slowly very much, but the lower bound is much better

second-round:0.012, 0.01, 0.008, 0.0005

| base_lr |best_accuracy|last_accuracy|information|
|---------|-|-|-|
|0.012|
|0.01|
|0.008|
|0.0005|


