import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_tea_caption import Tea_DataLoader as Tea_Caption_DataLoader
from dataloaders.dataloader_tea_retrieval import Tea_DataLoader



def dataloader_tea_train(args, tokenizer):
    if args.task_type == "retrieval":
        DataSet_DataLoader = Tea_DataLoader
    else:
        DataSet_DataLoader = Tea_Caption_DataLoader

    tea_dataset = DataSet_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(tea_dataset)
    dataloader = DataLoader(
        tea_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(tea_dataset), train_sampler

def dataloader_tea_test(args, tokenizer, subset="test"):
    # adding the dataloader test
    if args.task_type == "retrieval":
        DataSet_DataLoader = Tea_DataLoader
    else:
        DataSet_DataLoader = Tea_Caption_DataLoader
    tea_testset = DataSet_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )
    dataloader_tea = DataLoader(
        tea_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_tea, len(tea_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["tea"] = {"train":dataloader_tea_train, "val":dataloader_tea_test, "test":dataloader_tea_test}

