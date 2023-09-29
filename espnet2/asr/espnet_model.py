import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder,AVInDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder,AVOutEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr.transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)
from espnet2.asr.frontend.video_frontend import VideoFrontend
from espnet2.asr.preencoder.wav import WavPreEncoder

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        only_pdfloss: bool = False,
        pdfloss_skipencoder: bool = False,
        pdfloss_weight: float = 0.0,
        pdf_lsm_weigth: float = 0.0,
        pdf_cnum: int = 9024,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight+pdfloss_weight <= 1.0, f"ctc:{ctc_weight},pdf:{pdfloss_weight}"
        if interctc_weight != 0.0:
            logging.info(f"interctc_weight:{interctc_weight}")
        super().__init__()
        # NOTE (Shih-Lun): else case is for OpenAI Whisper ASR model,
        #                  which doesn't use <blank> token
        if sym_blank in token_list:
            self.blank_id = token_list.index(sym_blank)
        else:
            self.blank_id = 0
        if sym_sos in token_list:
            self.sos = token_list.index(sym_sos)
        else:
            self.sos = vocab_size - 1
        if sym_eos in token_list:
            self.eos = token_list.index(sym_eos)
        else:
            self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.aux_ctc = aux_ctc
        self.token_list = token_list.copy()
        self.only_pdfloss = only_pdfloss
        self.pdfloss_skipencoder = pdfloss_skipencoder
        self.pdfloss_weight = pdfloss_weight
        self.pdf_cnum = pdf_cnum

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        
        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if abs(1.0-self.ctc_weight-self.pdfloss_weight) <= 1e-5 or only_pdfloss == True:
                self.decoder = None
                logging.warning(f"Set decoder to none as ctc_weight=={self.ctc_weight},pdfloss_weight=={pdfloss_weight}")
            else:
                self.decoder = decoder
                

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0 or only_pdfloss == True:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__

        if self.is_encoder_whisper:
            assert (
                self.frontend is None
            ), "frontend should be None when using full Whisper model"

        if lang_token_id != -1:
            self.lang_token_id = torch.tensor([[lang_token_id]])
        else:
            self.lang_token_id = None

        if pdfloss_weight != 0.0 or only_pdfloss==True :
            if not pdfloss_skipencoder:
                self.pdfclass_linear  = torch.nn.Linear(encoder.output_size(), pdf_cnum)
            else: 
                self.pdfclass_linear  = torch.nn.Linear(frontend.output_size(), pdf_cnum)
            self.criterion_pdf = LabelSmoothingLoss(
                size=pdf_cnum,
                padding_idx=ignore_id,
                smoothing=pdf_lsm_weigth,
            )

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor=None,
        text_lengths: torch.Tensor=None,
        pdf: torch.Tensor=None,
        pdf_lengths: torch.Tensor=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        if  text!=None: #speech & text
            assert text_lengths.dim() == 1, text_lengths.shape
            assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
            ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
            text = text[:, : text_lengths.max()] # for data-parallel
            text[text == -1] = self.ignore_id
        if pdf!=None: # speech & pdf
            assert pdf_lengths.dim() == 1, pdf_lengths.shape
            assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == pdf.shape[0]
                == pdf_lengths.shape[0]
            ), (speech.shape, speech_lengths.shape, pdf.shape, pdf_lengths.shape)
            pdf = pdf[:, : pdf_lengths.max()]
            
        batch_size = speech.shape[0]       
            
        # 1. Encoder
        encoder_out, encoder_out_lens, frontend_out, frontend_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]     

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()
        # 2. Loss
        # 2.1. only_pdfloss
        if self.only_pdfloss:
            assert pdf != None, "pdf_weight:{self.pdfloss_weight} or check pdf input"
            loss,acc_pdf = self._calc_pdf_loss(
                frontend_out if self.pdfloss_skipencoder else encoder_out, pdf,
                )
          
            stats["loss_pdf"] = loss.detach() if loss is not None else None
            stats["acc_pdf"] = acc_pdf
        else:  
        # 2.2. CTC loss
            if self.ctc_weight != 0.0:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

                # Collect CTC branch stats
                stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
                stats["cer_ctc"] = cer_ctc

            # Intermediate CTC (optional)
            loss_interctc = 0.0
            if self.interctc_weight != 0.0 and intermediate_outs is not None:
                #interctc_weight interctc_use_conditioning
                for layer_idx, intermediate_out in intermediate_outs:
                    # we assume intermediate_out has the same length & padding
                    # as those of encoder_out

                    # use auxillary ctc data if specified
                    loss_ic = None
                    if self.aux_ctc is not None:
                        idx_key = str(layer_idx)
                        if idx_key in self.aux_ctc:
                            aux_data_key = self.aux_ctc[idx_key]
                            aux_data_tensor = kwargs.get(aux_data_key, None)
                            aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                            if aux_data_tensor is not None and aux_data_lengths is not None:
                                loss_ic, cer_ic = self._calc_ctc_loss(
                                    intermediate_out,
                                    encoder_out_lens,
                                    aux_data_tensor,
                                    aux_data_lengths,
                                )
                            else:
                                raise Exception(
                                    "Aux. CTC tasks were specified but no data was found"
                                )
                    if loss_ic is None:
                        loss_ic, cer_ic = self._calc_ctc_loss(
                            intermediate_out, encoder_out_lens, text, text_lengths
                        )
                    loss_interctc = loss_interctc + loss_ic

                    # Collect Intermedaite CTC stats
                    stats["loss_interctc_layer{}".format(layer_idx)] = (
                        loss_ic.detach() if loss_ic is not None else None
                    )
                    stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

                loss_interctc = loss_interctc / len(intermediate_outs)

                # calculate whole encoder loss
                loss_ctc = (
                    1 - self.interctc_weight
                ) * loss_ctc + self.interctc_weight * loss_interctc

            if self.use_transducer_decoder:
                # 2.3. Trasducer loss
                (
                    loss_transducer,
                    cer_transducer,
                    wer_transducer,
                ) = self._calc_transducer_loss(
                    encoder_out,
                    encoder_out_lens,
                    text,
                )

                if loss_ctc is not None:
                    loss = loss_transducer + (self.ctc_weight * loss_ctc)
                else:
                    loss = loss_transducer

                # Collect Transducer branch stats
                stats["loss_transducer"] = (
                    loss_transducer.detach() if loss_transducer is not None else None
                )
                stats["cer_transducer"] = cer_transducer
                stats["wer_transducer"] = wer_transducer

            else:
                # 2.4. pdf loss 
                if self.pdfloss_weight != 0.0:
                    assert pdf != None, "check pdf input"
                    loss_pdf,acc_pdf = self._calc_pdf_loss(
                        frontend_out if self.pdfloss_skipencoder else encoder_out, pdf,
                    )
                    stats["loss_pdf"] = loss_pdf.detach() if loss_pdf is not None else None
                    stats["acc_pdf"] = acc_pdf

                # 2.5. Attention loss
                if abs(1.0-self.ctc_weight-self.pdfloss_weight) >= 1e-5:
                    loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )
                    # Collect Attn branch stats
                    stats["loss_att"] = loss_att.detach() if loss_att is not None else None
                    stats["acc"] = acc_att
                    stats["cer"] = cer_att
                    stats["wer"] = wer_att

                #  2.6. weighted sum loss
                if self.pdfloss_weight == 1.0:
                    loss = loss_pdf
                elif self.ctc_weight == 1.0:
                    loss = loss_ctc
                elif self.pdfloss_weight == 0. and self.ctc_weight == 0.:
                    loss = loss_att
                elif self.pdfloss_weight == 0.0:
                    loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
                elif self.ctc_weight == 0.0:
                    loss = self.pdfloss_weight * loss_pdf + (1 - self.pdfloss_weight) * loss_att
                elif abs(self.pdfloss_weight + self.ctc_weight-1.0) <= 1e-5 :
                    loss = self.ctc_weight * loss_ctc + self.pdfloss_weight * loss_pdf
                else:
                    loss = self.ctc_weight * loss_ctc + self.pdfloss_weight * loss_pdf + (1.0-self.pdfloss_weight-self.ctc_weight) * loss_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        if not isinstance (self.preencoder,WavPreEncoder):
            with autocast(False):
                # 1. Extract feats
                feats, feats_lengths = self._extract_feats(speech, speech_lengths)

                # 2. Data augmentation
                if self.specaug is not None and self.training:
                    feats, feats_lengths = self.specaug(feats, feats_lengths)

                # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
                if self.normalize is not None:
                    feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 4. Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            if isinstance (self.preencoder,WavPreEncoder):
                feats, feats_lengths = self.preencoder(speech, speech_lengths)  #[B,T]->[B,T,D] 25fps
            else:
                feats, feats_lengths = self.preencoder(feats, feats_lengths) #[B,T,D]->[B,T,D] 100fps

        # 5. Forward encoder (Batch, Length, Dim) -> (Batch, Length2, Dim2)
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # 6. Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens, feats, feats_lengths

        return encoder_out, encoder_out_lens, feats, feats_lengths
 

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for multiple channels #B,T,channel_num-> channel_num,B,T,->channel_num*B,T
        if speech.dim() == 3:
            bsize,tlen,channel_num = speech.shape
            speech = speech.permute((2, 0, 1)).reshape(channel_num*bsize,tlen) 
            speech_lengths = speech_lengths.repeat(channel_num)

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_pdf_loss(
        self,
        encoder_out: torch.Tensor,
        ys_pad: torch.Tensor,
    ):  
        ys_head_pd = self.pdfclass_linear(encoder_out) #B,T,encodeoutdim -> B,T,class_num
        
        #soft align for some pdf and label
        tag_len = ys_pad.shape[1]
        hyp_len = ys_head_pd.shape[1]
        # logging.info(f"ys_pad_shape:{ys_pad.shape},ys_head_pad_shape:{ys_head_pd.shape}")
        if  tag_len != hyp_len:
            if max(tag_len/hyp_len,hyp_len/tag_len) > 3:
                raise ValueError(f"ys_pad_shape:{ys_pad.shape},ys_head_pad_shape:{ys_head_pd.shape}")
            else:
                cutlen = min(tag_len,hyp_len)
                ys_head_pd = ys_head_pd[:,:cutlen] #[B,T,C]
                ys_pad = ys_pad[:,:cutlen] #[B,T]
                
        loss_pdf = self.criterion_pdf(ys_head_pd, ys_pad) #ignore_id = -1
        
        acc_pdf = th_accuracy(
                ys_head_pd.reshape(-1, self.pdf_cnum),
                ys_pad,
                ignore_label=self.ignore_id,
            )
        return loss_pdf, acc_pdf

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):  
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1
            
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer


class ESPnetAVSRModel(ESPnetASRModel):  
    
    def __init__(
                self,
                vocab_size: int,
                token_list: Union[Tuple[str, ...], List[str]],
                frontend: Optional[AbsFrontend],
                video_frontend : VideoFrontend,
                specaug: Optional[AbsSpecAug],
                normalize: Optional[AbsNormalize],
                preencoder: Optional[AbsPreEncoder],
                encoder: AbsEncoder,
                postencoder: Optional[AbsPostEncoder],
                decoder: AbsDecoder,
                ctc: CTC,
                joint_network: Optional[torch.nn.Module],
                ctc_weight: float = 0.5,
                ignore_id: int = -1,
                lsm_weight: float = 0.0,
                length_normalized_loss: bool = False,
                report_cer: bool = True,
                report_wer: bool = True,
                sym_space: str = "<space>",
                sym_blank: str = "<blank>",
                extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super(ESPnetASRModel, self).__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.video_frontend = video_frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def encode(
        self, 
        speech: torch.Tensor, 
        speech_lengths: torch.Tensor,
        video: torch.Tensor,
        video_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """

        #1. video pre-encoder  (B, T, 96, 96,3) -> (B,T,D)  25ps 
        if self.video_frontend is not None:
            video,video_lengths = self.video_frontend(video,video_lengths)

        #2. STFT+AG+Norm
        #WavPreEncoder is 1-convd and resnet 1d based N,T->N,T,C
        #WavPreEncoder only onput accept waveform , don't have to do STSF 
        if not isinstance (self.preencoder,WavPreEncoder):
            with autocast(False):
                # a. Extract feats
                feats, feats_lengths = self._extract_feats(speech, speech_lengths)
                # b. Data augmentation
                if self.specaug is not None and self.training:
                        feats, feats_lengths = self.specaug(feats, feats_lengths)
                
                # c. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
                if self.normalize is not None:
                    feats, feats_lengths = self.normalize(feats, feats_lengths)
                    
                    
        #3. Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            if isinstance (self.preencoder,WavPreEncoder):
                feats, feats_lengths = self.preencoder(speech, speech_lengths)  #[B,T]->[B,T,D] 25fps
            else:
                feats, feats_lengths = self.preencoder(feats, feats_lengths) #[B,T,D]->[B,T,D] 100fps
              
        # 4. soft alignment for WavPreEncoder 
        # for wav preencoder auido 25ps, video 25ps; for feat preencoder audio 100 ps ,video 25 ps 
        # align av frames if both audio and video nearly close to 25ps or alignment , if 4*video = audio don't alignment
        if isinstance (self.preencoder,WavPreEncoder):
            if not feats_lengths.equal(video_lengths):
                if (feats_lengths-video_lengths).abs().sum() < (feats_lengths-video_lengths*4).abs().sum():
                    feats_lengths = feats_lengths.min(video_lengths)
                    video_lengths = feats_lengths.clone()
                    feats = feats[:,:max(feats_lengths)]
                    video = video[:,:max(feats_lengths)]
        
        # 5. Encoder
        # 5.1. for encoders which only output audio memories
        if not isinstance(self.encoder,AVOutEncoder):
            encoder_out, encoder_out_lens, _ = self.encoder(feats,feats_lengths,video,video_lengths)#[B,T,D]
    
            # Post-encoder, e.g. NLU
            if self.postencoder is not None:
                encoder_out, encoder_out_lens = self.postencoder(
                    encoder_out, encoder_out_lens
                )

            assert encoder_out.size(0) == speech.size(0), (
                encoder_out.size(),
                speech.size(0),
            )
            assert encoder_out.size(1) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

            return encoder_out, encoder_out_lens
        
        # 5.2. for encoders which output audio and video memories
        else:
            feats,feats_lengths,video,video_lengths,_ = self.encoder(feats,feats_lengths,video,video_lengths)#[B,T,D]
            assert feats.size(0)==video.size(0)==speech.size(0), (
                    feats.size(),
                    video.size(),
                    speech.size(0),
                )
            assert feats.size(1) <= feats_lengths.max() and video.size(1) <= video_lengths.max(), (
                feats.size(),
                feats_lengths.max(),
                video.size(),
                video_lengths.max(),
            )

            return feats,feats_lengths,video,video_lengths

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        video: torch.Tensor,
        video_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == video.shape[0]
            == video_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape,video.shape,text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        if not isinstance(self.encoder,AVOutEncoder): 
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths,video,video_lengths)
        else:
            encoder_out,encoder_out_lens,encoder_vout,encoder_vout_lens = self.encode(speech, speech_lengths,video,video_lengths)
        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 2. loss
        # 2.1. CTC loss
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
                # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        if self.use_transducer_decoder:
            # 2.2. Transducer loss
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2.3. Attention loss
            if self.ctc_weight != 1.0:
                if not isinstance(self.decoder,AVInDecoder):
                    loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )
                else:
                    loss_att, acc_att, cer_att, wer_att = self._calc_avin_att_loss(
                        encoder_out, encoder_out_lens,encoder_vout,encoder_vout_lens,text, text_lengths
                    )

            # 2.4. CTC-Att loss 
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att
            
            

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    # for decoder input audio and video memeories
    def _calc_avin_att_loss(
        self,
        feats: torch.Tensor,
        feats_lens: torch.Tensor,
        video: torch.Tensor,
        video_lengths: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):  
    
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            feats, feats_lens, video, video_lengths, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    
class ESPnetVSRModel(ESPnetASRModel):  
    
    def __init__(
                self,
                vocab_size: int,
                token_list: Union[Tuple[str, ...], List[str]],
                video_frontend : VideoFrontend,
                encoder: AbsEncoder,
                decoder: AbsDecoder,
                ctc: CTC,
                joint_network: Optional[torch.nn.Module],
                ctc_weight: float = 0.5,
                ignore_id: int = -1,
                lsm_weight: float = 0.0,
                length_normalized_loss: bool = False,
                report_cer: bool = True,
                report_wer: bool = True,
                sym_space: str = "<space>",
                sym_blank: str = "<blank>",
                extract_feats_in_collect_stats: bool = True,
                only_pdfloss: bool = False,
                pdfloss_skipencoder: bool = False,
                pdfloss_weight: float = 0.0,
                pdf_lsm_weigth: float = 0.0,
                pdf_cnum: int = 9024,

    ): 
        assert check_argument_types()
        assert 0.0 <= ctc_weight+pdfloss_weight <= 1.0, f"ctc:{ctc_weight},pdf:{pdfloss_weight}"
         
        super(ESPnetASRModel, self).__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()
        self.only_pdfloss = only_pdfloss
        self.pdfloss_skipencoder = pdfloss_skipencoder
        self.pdfloss_weight = pdfloss_weight
        self.pdf_cnum = pdf_cnum

        self.video_frontend = video_frontend
        self.encoder = encoder

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        
        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        if abs(1.0-self.ctc_weight-self.pdfloss_weight) <= 1e-5 or only_pdfloss == True:
            self.decoder = None
        else:
            self.decoder = decoder
            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )


        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )

        if ctc_weight == 0.0 or only_pdfloss == True:
            self.ctc = None
        else:
            self.ctc = ctc
        
        if pdfloss_weight != 0.0 or only_pdfloss==True :
            if not pdfloss_skipencoder:
                self.pdfclass_linear  = torch.nn.Linear(encoder.output_size(), pdf_cnum)
            else: 
                self.pdfclass_linear  = torch.nn.Linear(video_frontend.output_size(), pdf_cnum)
            self.criterion_pdf = LabelSmoothingLoss(
                size=pdf_cnum,
                padding_idx=ignore_id,
                smoothing=pdf_lsm_weigth,
            )

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
    
    def encode(
        self, 
        video: torch.Tensor,
        video_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        #video pre-encoder (B, T, 96, 96,3) -> (B,T,D)  25ps 
        video,video_lengths = self.video_frontend(video,video_lengths)
        encoder_res = self.encoder(video,video_lengths)#[B,T,D]
        encoder_out, encoder_out_lens = encoder_res[0],encoder_res[1]
    
        assert encoder_out.size(0) == video.size(0), (
            encoder_out.size(),
            video.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens,video,video_lengths

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        video: torch.Tensor,
        video_lengths: torch.Tensor,
        pdf: torch.Tensor=None,
        pdf_lengths: torch.Tensor=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """ 
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            text.shape[0]
            == text_lengths.shape[0]
            == video.shape[0]
            == video_lengths.shape[0]
        ), (text.shape, text_lengths.shape,video.shape,text_lengths.shape)
        batch_size = video.shape[0]
        

        # for data-parallel
        text = text[:, : text_lengths.max()]

        if pdf != None :
            assert pdf.shape[0] == pdf_lengths.shape[0] == text.shape[0]
            pdf = pdf[:, : pdf_lengths.max()]

   
        # 1. Encoder
        encoder_out, encoder_out_lens, frontend_out, frontend_out_lens  = self.encode(video,video_lengths)
        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_pdf, acc_pdf = None, None
        stats = dict()
        # 2. loss
        # 2.1. only_pdfloss
        if self.only_pdfloss:
            assert pdf != None, "pdf_weight:{self.pdfloss_weight} or check pdf input"
            loss,acc_pdf = self._calc_pdf_loss(
                frontend_out if self.pdfloss_skipencoder else encoder_out, pdf,
                )
            stats["loss_pdf"] = loss.detach() if loss is not None else None
            stats["acc_pdf"] = acc_pdf
         
        else:
            # 2.2. ctc loss
            if self.ctc_weight != 0.0:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
                # Collect CTC branch stats
                stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
                stats["cer_ctc"] = cer_ctc

            # 2.2. pdf loss
            if self.pdfloss_weight != 0.0:
                assert pdf != None, "check pdf input"
                loss_pdf,acc_pdf = self._calc_pdf_loss(
                    frontend_out if self.pdfloss_skipencoder else encoder_out, pdf,
                )
                stats["loss_pdf"] = loss_pdf.detach() if loss_pdf is not None else None
                stats["acc_pdf"] = acc_pdf

            # 2.3. attention loss
            if abs(1.0-self.ctc_weight-self.pdfloss_weight) >= 1e-5:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
                # Collect Attention branch stats
                stats["loss_att"] = loss_att.detach() if loss_att is not None else None
                stats["acc"] = acc_att
                stats["cer"] = cer_att
                stats["wer"] = wer_att

            # 2.4. weight sum loss
            if self.pdfloss_weight == 1.0:
                loss = loss_pdf
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            elif self.pdfloss_weight == 0. and self.ctc_weight == 0.:
                loss = loss_att
            elif self.pdfloss_weight == 0.0:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
            elif self.ctc_weight == 0.0:
                loss = self.pdfloss_weight * loss_pdf + (1 - self.pdfloss_weight) * loss_att
            elif abs(self.pdfloss_weight + self.ctc_weight-1.0) <= 1e-5 :
                loss = self.ctc_weight * loss_ctc + self.pdfloss_weight * loss_pdf
            else:
                loss = self.ctc_weight * loss_ctc + self.pdfloss_weight * loss_pdf + (1.0-self.pdfloss_weight-self.ctc_weight) * loss_att
                
            

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight


