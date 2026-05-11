"""
Live-prompt residual variant that expects exact-width embeddings from the
embedding server and therefore skips adapter bootstrapping entirely.
"""

from config.configurator import configs
from models.general_cf.mult_vae_mddm_residual import mult_vae_MDDM_Residual


class mult_vae_MDDM_Residual_Live_Prompt_Direct(mult_vae_MDDM_Residual):
    def __init__(self, data_handler):
        super(mult_vae_MDDM_Residual_Live_Prompt_Direct, self).__init__(data_handler)

        # Force the trainer down the strict-dimension path. It will request
        # embedding_dimensions from the server and reject any mismatch.
        self.use_native_live_prompt_embeddings = False
        self._validate_live_prompt_dimensions()

    def _validate_live_prompt_dimensions(self):
        user_dim = int(self.usrprf_embeds_raw.shape[1])
        item_dim = int(self.itmprf_embeds_raw.shape[1])
        if user_dim != item_dim:
            raise ValueError(
                'Direct live prompt model requires matching user/item semantic widths, got {} and {}.'.format(
                    user_dim,
                    item_dim,
                )
            )

        configured_dim = configs.get('live_prompting', {}).get('embedding_dimensions')
        if configured_dim is None:
            return

        configured_dim = int(configured_dim)
        if configured_dim != user_dim:
            raise ValueError(
                'Direct live prompt model expects live_prompting.embedding_dimensions to match the base semantic width ({}), got {}.'.format(
                    user_dim,
                    configured_dim,
                )
            )