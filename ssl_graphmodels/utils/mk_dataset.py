from torchtext.legacy import data
from torchtext.legacy.datasets import DBpedia, YelpReviewPolarity, AmazonReviewFull


TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)


root_path = '/data/project/yinhuapark/DATA_RAW/torch_text'
YelpReviewPolaAmazonReviewFull(root=root_path)