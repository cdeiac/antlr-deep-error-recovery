'''
 for epoch in range(EPOCHS):
     train_loss, train_acc = train_one_epoch_with_windowing(
         source_windowed,
         target_windowed,
         padded_target,
         model,
         loss_function,
         optimizer,
         stride
     )
     epoch_train_loss.append(train_loss)
     epoch_train_acc.append(train_acc)
     valid_loss, valid_acc = evaluate_one_epoch_with_windowing(
         test_source_windowed,
         test_target_windowed,
         padded_test_target,
         model,
         loss_function,
         scheduler,
         stride
     )
     epoch_valid_loss.append(valid_loss)
     epoch_valid_acc.append(valid_acc)

     #predict_with_sliding_window(model, test_source_windowed[0], test_target_windowed[0], index2token)
     #predict_with_sliding_window(model, test_source_windowed[1], test_target_windowed[1], index2token)
     #predict_with_sliding_window(model, test_source_windowed[2], test_target_windowed[2], index2token)

     print(
         f'{datetime.now()} - Epoch {epoch + 1}: train_acc = {epoch_train_acc[epoch]}, train_loss = {epoch_train_loss[epoch]}, '
         f'valid_acc = {epoch_valid_acc[epoch]}, valid_loss = {epoch_valid_loss[epoch]}'
     )
 '''

'''
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(
        train_seqs,
        train_target_seqs,
        model,
        loss_function,
        optimizer
    )
    epoch_train_loss.append(train_loss)
    epoch_train_acc.append(train_acc)
    valid_loss, valid_acc = evaluate_one_epoch(
        test_seqs,
        test_target_seqs,
        model,
        loss_function,
        scheduler
    )
    epoch_valid_loss.append(valid_loss)
    epoch_valid_acc.append(valid_acc)

    predict(model, test_seqs[0], test_target_seqs[0], index2token)
    predict(model, test_seqs[1], test_target_seqs[1], index2token)

    #predict_with_sliding_window(model, test_source_windowed[1], test_target_windowed[1], index2token)
    #predict_with_sliding_window(model, test_source_windowed[2], test_target_windowed[2], index2token)

    print(
        f'{datetime.now()} - Epoch {epoch + 1}: train_acc = {epoch_train_acc[epoch]}, train_loss = {epoch_train_loss[epoch]}, '
        f'valid_acc = {epoch_valid_acc[epoch]}, valid_loss = {epoch_valid_loss[epoch]}'
    )
'''
