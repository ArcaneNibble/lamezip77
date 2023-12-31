extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Expr, Ident, Path, Token,
};

struct MacroInput {
    id: Ident,
    _comma1: Token![,],
    outp: Expr,
    _comma2: Option<Token![,]>,
    crate_path: Option<Path>,
}

impl Parse for MacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let id = input.parse()?;
        let _comma1 = input.parse()?;
        let outp = input.parse()?;
        let maybe_comma: Option<Token![,]> = input.parse()?;
        if maybe_comma.is_none() {
            Ok(Self {
                id,
                _comma1,
                outp,
                _comma2: None,
                crate_path: None,
            })
        } else {
            Ok(Self {
                id,
                _comma1,
                outp,
                _comma2: None,
                crate_path: Some(input.parse()?),
            })
        }
    }
}

#[proc_macro]
pub fn nintendo_lz_decompress_make(tokens: TokenStream) -> TokenStream {
    let input = parse_macro_input!(tokens as MacroInput);

    let ident = input.id;
    let outp = input.outp;
    let crate_ref = if let Some(crate_path) = input.crate_path {
        quote! { #crate_path }
    } else {
        quote! { ::lamezip77 }
    };

    quote! {
        let __lamezip77_hidden_inner_state = #crate_ref::StreamingDecompressInnerState::<4>::new();
        let __lamezip77_hidden_inner_future = ::core::pin::pin!(#crate_ref::nintendo_lz::decompress_impl(
            #outp,
            __lamezip77_hidden_inner_state.get_peeker::<1>(),
            __lamezip77_hidden_inner_state.get_peeker::<2>(),
            __lamezip77_hidden_inner_state.get_peeker::<4>(),
        ));
        let mut #ident = #crate_ref::nintendo_lz::Decompress::new(&__lamezip77_hidden_inner_state, __lamezip77_hidden_inner_future);
    }
    .into()
}

#[proc_macro]
pub fn fastlz_decompress_make(tokens: TokenStream) -> TokenStream {
    let input = parse_macro_input!(tokens as MacroInput);

    let ident = input.id;
    let outp = input.outp;
    let crate_ref = if let Some(crate_path) = input.crate_path {
        quote! { #crate_path }
    } else {
        quote! { ::lamezip77 }
    };

    quote! {
        let __lamezip77_hidden_inner_state = #crate_ref::StreamingDecompressInnerState::<2>::new();
        let __lamezip77_hidden_inner_future = ::core::pin::pin!(#crate_ref::fastlz::decompress_impl(
            #outp,
            __lamezip77_hidden_inner_state.get_peeker::<1>(),
            __lamezip77_hidden_inner_state.get_peeker::<2>(),
        ));
        let mut #ident = #crate_ref::fastlz::Decompress::new(&__lamezip77_hidden_inner_state, __lamezip77_hidden_inner_future);
    }
    .into()
}

#[proc_macro]
pub fn lz4_decompress_make(tokens: TokenStream) -> TokenStream {
    let input = parse_macro_input!(tokens as MacroInput);

    let ident = input.id;
    let outp = input.outp;
    let crate_ref = if let Some(crate_path) = input.crate_path {
        quote! { #crate_path }
    } else {
        quote! { ::lamezip77 }
    };

    quote! {
        let __lamezip77_hidden_inner_state = #crate_ref::StreamingDecompressInnerState::<2>::new();
        let __lamezip77_hidden_inner_future = ::core::pin::pin!(#crate_ref::lz4::decompress_impl(
            #outp,
            __lamezip77_hidden_inner_state.get_peeker::<1>(),
            __lamezip77_hidden_inner_state.get_peeker::<2>(),
        ));
        let mut #ident = #crate_ref::lz4::Decompress::new(&__lamezip77_hidden_inner_state, __lamezip77_hidden_inner_future);
    }
    .into()
}

#[proc_macro]
pub fn deflate_decompress_make(tokens: TokenStream) -> TokenStream {
    let input = parse_macro_input!(tokens as MacroInput);

    let ident = input.id;
    let outp = input.outp;
    let crate_ref = if let Some(crate_path) = input.crate_path {
        quote! { #crate_path }
    } else {
        quote! { ::lamezip77 }
    };

    quote! {
        let __lamezip77_hidden_inner_state = #crate_ref::StreamingDecompressInnerState::<2>::new();
        let __lamezip77_hidden_inner_future = ::core::pin::pin!(#crate_ref::deflate::decompress_impl(
            #outp,
            __lamezip77_hidden_inner_state.get_peeker::<1>(),
            __lamezip77_hidden_inner_state.get_peeker::<2>(),
        ));
        let mut #ident = #crate_ref::deflate::Decompress::new(&__lamezip77_hidden_inner_state, __lamezip77_hidden_inner_future);
    }
    .into()
}
