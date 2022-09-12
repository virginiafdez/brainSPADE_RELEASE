'''
Things that we tried and didn't work out.
'''
# ABOUT Intensify disease: increase focus of the discriminator in the diseased tissue.
# IN Compute generator loss
# if self.opt.intensify_disease:
#     disease_mask = uvir.retrieve_disease_mask(input_semantics)[topk_indices, ...]
#     for j_ind, j in enumerate(pred_fake_GAN):
#         semantic_map_int = nnf.interpolate(disease_mask, size=(j[-1].shape[-2], j[-1].shape[-1]),
#                                            mode='bicubic', align_corners=False).type(torch.bool)
#         pred_fake_GAN[j_ind][-1][~semantic_map_int] *= 0.001

# IN Compute discriminator loss
# if self.opt.intensify_disease:
#     disease_mask = uvir.retrieve_disease_mask(input_semantics)[topk_indices, ...]
#     for j_ind, j in enumerate(pred_real_GAN):
#         semantic_map_int = nnf.interpolate(disease_mask, size=(j[-1].shape[-2], j[-1].shape[-1]),
#                                            mode='bicubic', align_corners=False).type(torch.bool)
#
#         pred_real_GAN[j_ind][-1][~semantic_map_int] *= 0.001  # Almost zero non disease
#         pred_fake_GAN[j_ind][-1][~semantic_map_int] *= 0.001
# Options:
# parser.add_argument('--intensify_disease', action='store_true', help="When calculating the adversarial loss,"
#                                                                      "give more importance to disease")
#
