from pycorrector import MacBertCorrector
m = MacBertCorrector()
a = m.tokenizer
print(a)
print(m.correct_batch(['今天新情很好', '你找到你最喜欢的工作，我也很高心。']))