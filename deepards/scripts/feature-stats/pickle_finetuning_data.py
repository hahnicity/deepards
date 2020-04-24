from deepards.dataset import ARDSRawDataset

train_dataset = ARDSRawDataset(
                '/home/bhargav/deepards-data-finetuning/ardsdetection_data_anon_non_consent_filtered/',
                1,
                '/home/bhargav/deepards-data-finetuning/ardsdetection_data_anon_non_consent_filtered/cohort-description.csv',
                100,
                'padded_breath_by_breath_with_custom_bm_target',
                ['iTime', 'eTime', 'inst_RR', 'mean_flow_from_pef', 'I:E ratio', 'tve:tvi ratio'],
                to_pickle='~/deepards-data-finetuning/data-with-bm-features.pkl',
                kfold_num=None,
                total_kfolds=None,
                unpadded_downsample_factor=4,
                drop_frame_if_frac_missing=False,
                oversample_minority=True,
            )


