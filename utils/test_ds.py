from dataset import *
from tqdm import tqdm

# Using ItemSet for testing makes spotting bugs easier
item_set = AnyItemSet([5, 5, 5], intentional_distractors=3,  sample_scaling=2,
                      upsample=True, balanced_distractors=True)
train_ds, validate_ds, test_ds = item_set.get_zero_shot_datasets((0.6, .4))


for object_idx in tqdm(range(len(item_set.objects))):
    for sample_idx in range(item_set.sample_scaling):
        for intend_grouped_by_abstraction in item_set.intentions:
            for intend in intend_grouped_by_abstraction:
                (sender_object, intention), label, receiver_input = item_set.get_item(object_idx, intend, lambda x:x)

                for i,target_object in enumerate(receiver_input):
                    if i is label:
                        # Check if target is correct
                        for (j, item), is_any in zip(target_object.items(), intention):
                            if not is_any:
                                assert sender_object.iloc[j] == item

                    else:
                        # Check if distractor is correct
                        distractor_is_different = False
                        for (j, item), is_any in zip(target_object.items(), intention):
                            if not is_any:
                                if sender_object.iloc[j] != item:
                                    distractor_is_different = True

                        if not distractor_is_different:
                            raise AssertionError("Distractor could be a target")


print("\nNo assertions errors occurred. Everything works fine.")
