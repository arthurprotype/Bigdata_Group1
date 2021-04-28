result = model.predict (X_test)
test_num = result.shape[0]
result_sort = np.argsort (result, axis=1)
right, ictal_r, pre_ictal, ictal_n = 0, 0, 0, 0
for i in range (test_num):
    if result_sort[i][-1] == 2:
        pre_ictal += 1
        if result_sort[i][-1] == list (y_test[i]).index (1):
            ictal_r += 1
    if list (y_test[i]).index (1) == 2:
        ictal_n += 1
    if result_sort[i][-1] == list (y_test[i]).index (1):
        right += 1
print ("acc: %.4f" % (right / test_num))
print ("sen: %.4f" % (ictal_r / ictal_n))
print ("recall: %.4f" % (ictal_r / pre_ictal))