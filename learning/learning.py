# 学習させる(for)
def execution(num, sess, x, p, t, loss, train_step, correct_prediction, accuracy, train_x, train_t, y):
    i = 0
    for _ in range(num):
        i += 1
        sess.run(train_step, feed_dict={x:train_x, t:train_t})
        if i % 100 == 0:
            loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:train_x, t:train_t})
            result = sess.run(y, feed_dict={x: train_x})
            print ('Step: %d, Loss: %f, Accuracy: %f,'% (i, loss_val, acc_val))
            print(str(result[0]))
    
    
