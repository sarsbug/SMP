* SMP
#Pytorch implementation for Deep Self-Learning From Noisy Labels

个人实现的SMP算法，测试集使用的是fashion-mnist，分别进行了symmetric测试和asymmetric测试，发现结果不够稳定，对于asymmetric noisy label的处理无效。代码可能有不完善的地方，欢迎交流指正。

##symmetric测试
y_pseudo_symmetric_noise

###不使用SMP算法

Counter({9: 7115, 6: 7072, 4: 7024, 5: 7019, 1: 7010, 2: 6985, 3: 6979, 8: 6957, 0: 6946, 7: 6893})

 original y_pseudo acc:0.595114
 
*  Epoch   1: Classification Loss: 0.007275 acc: 0.806871
*  Epoch   2: Classification Loss: 0.006925 acc: 0.825529
*  Epoch   3: Classification Loss: 0.006794 acc: 0.835200
*  Epoch   4: Classification Loss: 0.006706 acc: 0.856143
*  Epoch   5: Classification Loss: 0.006624 acc: 0.860571
*  Epoch   6: Classification Loss: 0.006562 acc: 0.867857
*  Epoch   7: Classification Loss: 0.006500 acc: 0.869429
*  Epoch   8: Classification Loss: 0.006440 acc: 0.877929
*  Epoch   9: Classification Loss: 0.006368 acc: 0.881071
*  Epoch  10: Classification Loss: 0.006302 acc: 0.885343
*  Epoch  11: Classification Loss: 0.006222 acc: 0.883686
*  Epoch  12: Classification Loss: 0.006164 acc: 0.883286
*  Epoch  13: Classification Loss: 0.006069 acc: 0.879514
*  Epoch  14: Classification Loss: 0.005965 acc: 0.880771
*  Epoch  15: Classification Loss: 0.005879 acc: 0.863400
*  Epoch  16: Classification Loss: 0.005770 acc: 0.865986
*  Epoch  17: Classification Loss: 0.005651 acc: 0.844100
*  Epoch  18: Classification Loss: 0.005510 acc: 0.849271
*  Epoch  19: Classification Loss: 0.005391 acc: 0.832271
*  Epoch  20: Classification Loss: 0.005234 acc: 0.828543
*  
 final acc:0.829

###使用SMP算法
Counter({9: 7115, 6: 7072, 4: 7024, 5: 7019, 1: 7010, 2: 6985, 3: 6979, 8: 6957, 0: 6946, 7: 6893})

original y_pseudo acc:0.595114

*  Epoch   1: Classification Loss: 0.007275 acc: 0.823900
*  Epoch   2: Classification Loss: 0.006914 acc: 0.818414
*  Epoch   3: Classification Loss: 0.006760 acc: 0.844757
*  Epoch   4: Classification Loss: 0.006677 acc: 0.853986
*  Epoch   5: Classification Loss: 0.006603 acc: 0.853443
* with smp Epoch   6: Classification Loss: 0.008154 acc: 0.860429
* with smp Epoch   7: Classification Loss: 0.008165 acc: 0.876100
* with smp Epoch   8: Classification Loss: 0.008178 acc: 0.869971
* with smp Epoch   9: Classification Loss: 0.008209 acc: 0.885157
* with smp Epoch  10: Classification Loss: 0.008201 acc: 0.884886
* with smp Epoch  11: Classification Loss: 0.008213 acc: 0.886371
* with smp Epoch  12: Classification Loss: 0.008185 acc: 0.880629
* with smp Epoch  13: Classification Loss: 0.008191 acc: 0.886529
* with smp Epoch  14: Classification Loss: 0.008096 acc: 0.876386
* with smp Epoch  15: Classification Loss: 0.008066 acc: 0.891257
* with smp Epoch  16: Classification Loss: 0.008071 acc: 0.877500
* with smp Epoch  17: Classification Loss: 0.008153 acc: 0.896029
* with smp Epoch  18: Classification Loss: 0.008122 acc: 0.896443
* with smp Epoch  19: Classification Loss: 0.008006 acc: 0.885543
* with smp Epoch  20: Classification Loss: 0.007960 acc: 0.888086
* 
 final acc:0.888


##asymmetric测试

y_pseudo_asymmetric_noise

###不使用SMP算法

Counter({2: 11200, 1: 7000, 3: 7000, 4: 7000, 5: 7000, 6: 7000, 7: 7000, 8: 7000, 9: 7000, 0: 2800})

 original y_pseudo acc:0.650000
 
*  Epoch   1: Classification Loss: 0.003714 acc: 0.534043
*  Epoch   2: Classification Loss: 0.002870 acc: 0.556986
*  Epoch   3: Classification Loss: 0.002731 acc: 0.622929
*  Epoch   4: Classification Loss: 0.002640 acc: 0.571757
*  Epoch   5: Classification Loss: 0.002557 acc: 0.614829
*  Epoch   6: Classification Loss: 0.002491 acc: 0.654057
*  Epoch   7: Classification Loss: 0.002433 acc: 0.658071
*  Epoch   8: Classification Loss: 0.002379 acc: 0.607929
*  Epoch   9: Classification Loss: 0.002328 acc: 0.580900
*  Epoch  10: Classification Loss: 0.002286 acc: 0.696757
*  Epoch  11: Classification Loss: 0.002217 acc: 0.647386
*  Epoch  12: Classification Loss: 0.002156 acc: 0.615586
*  Epoch  13: Classification Loss: 0.002111 acc: 0.649986
*  Epoch  14: Classification Loss: 0.002039 acc: 0.597586
*  Epoch  15: Classification Loss: 0.001983 acc: 0.648257
*  Epoch  16: Classification Loss: 0.001914 acc: 0.626500
*  Epoch  17: Classification Loss: 0.001877 acc: 0.626929
*  Epoch  18: Classification Loss: 0.001801 acc: 0.620700
*  Epoch  19: Classification Loss: 0.001750 acc: 0.632800
*  Epoch  20: Classification Loss: 0.001664 acc: 0.633586
*  
 final acc:0.634

###使用SMP算法

 original y_pseudo acc:0.650000
 
*  Epoch   1: Classification Loss: 0.003894 acc: 0.599229
*  Epoch   2: Classification Loss: 0.002946 acc: 0.564500
*  Epoch   3: Classification Loss: 0.002769 acc: 0.572757
*  Epoch   4: Classification Loss: 0.002683 acc: 0.612086
*  Epoch   5: Classification Loss: 0.002586 acc: 0.577857
* with smp Epoch   6: Classification Loss: 0.007546 acc: 0.593029
* with smp Epoch   7: Classification Loss: 0.007269 acc: 0.621286
* with smp Epoch   8: Classification Loss: 0.007338 acc: 0.583386
* with smp Epoch   9: Classification Loss: 0.007244 acc: 0.586643
* with smp Epoch  10: Classification Loss: 0.007313 acc: 0.617457
* with smp Epoch  11: Classification Loss: 0.007343 acc: 0.625671
* with smp Epoch  12: Classification Loss: 0.007210 acc: 0.639771
* with smp Epoch  13: Classification Loss: 0.007220 acc: 0.590057
* with smp Epoch  14: Classification Loss: 0.007224 acc: 0.640129
* with smp Epoch  15: Classification Loss: 0.007340 acc: 0.618157
* with smp Epoch  16: Classification Loss: 0.007226 acc: 0.593586
* with smp Epoch  17: Classification Loss: 0.006971 acc: 0.648157
* with smp Epoch  18: Classification Loss: 0.007242 acc: 0.617571
* with smp Epoch  19: Classification Loss: 0.007192 acc: 0.597871
* with smp Epoch  20: Classification Loss: 0.007241 acc: 0.604500
* 
 final acc:0.605

