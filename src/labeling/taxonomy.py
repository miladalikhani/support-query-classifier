"""Banking77 class descriptions used in the teacher prompt.

Class **names** are owned by the dataset (`src.data.banking77` exposes them via
`id_to_label`). This module owns the human-readable **descriptions** that go
into the teacher prompt so Gemini can distinguish ambiguous pairs (e.g.,
`pending_transfer` vs `transfer_not_received_by_recipient`).

Quality of these descriptions is validated empirically by [P3][T5]
(teacher accuracy on the golden subset). If accuracy is low, iterate here.
"""

BANKING77_DESCRIPTIONS: dict[str, str] = {
    "activate_my_card": "Customer wants to activate a new or replacement card they have received.",
    "age_limit": "Customer asks about the minimum age requirement to open or use an account.",
    "apple_pay_or_google_pay": "Questions or issues about using Apple Pay or Google Pay (mobile wallets) with the card, including troubleshooting.",
    "atm_support": "Asks where to find compatible ATMs or about ATM compatibility in general.",
    "automatic_top_up": "Customer wants to set up, change, or ask about automatic top-up rules.",
    "balance_not_updated_after_bank_transfer": "Bank transfer completed but the balance does not yet reflect the change.",
    "balance_not_updated_after_cheque_or_cash_deposit": "Cash or cheque was deposited but the balance has not yet updated.",
    "beneficiary_not_allowed": "Customer has trouble adding or transferring to/from a beneficiary, including incoming transfers being declined due to beneficiary rules.",
    "cancel_transfer": "Customer wants to cancel or reverse a transfer they have already initiated.",
    "card_about_to_expire": "The card is approaching its expiry date and the customer asks what happens next.",
    "card_acceptance": "Asks whether the card is accepted at a specific merchant, country, or for a specific purchase.",
    "card_arrival": "Customer is asking about the status of a card that has been ordered but has not yet arrived.",
    "card_delivery_estimate": "Asks how long card delivery will take or whether expedited delivery is possible.",
    "card_linking": "Customer wants to link the card to another account, service, or device.",
    "card_not_working": "The card is failing to make payments or transactions in general.",
    "card_payment_fee_charged": "Customer was charged an unexpected fee on a card payment.",
    "card_payment_not_recognised": "Customer sees a card payment on their statement that they do not recognize or did not make.",
    "card_payment_wrong_exchange_rate": "Customer believes the exchange rate applied to a specific card payment was incorrect.",
    "card_swallowed": "An ATM has retained or 'swallowed' the customer's card.",
    "cash_withdrawal_charge": "Customer asks about fees charged for cash withdrawals.",
    "cash_withdrawal_not_recognised": "Customer sees an ATM/cash withdrawal on their statement that they did not make.",
    "change_pin": "Customer wants to change the PIN on their card.",
    "compromised_card": "Customer suspects their card details have been stolen or used fraudulently while the physical card is still in their possession.",
    "contactless_not_working": "Contactless payments are failing while chip/PIN may still work.",
    "country_support": "Asks whether the service operates in, or supports, a specific country.",
    "declined_card_payment": "A card payment was declined and the customer wants to know why.",
    "declined_cash_withdrawal": "A cash withdrawal attempt was declined by the bank or ATM.",
    "declined_transfer": "A transfer attempt was declined by the bank.",
    "direct_debit_payment_not_recognised": "Customer sees a direct debit on their statement that they do not recognize.",
    "disposable_card_limits": "Asks about limits (e.g., spending, count) on disposable virtual cards.",
    "edit_personal_details": "Customer wants to update personal information such as name, address, email, or phone number.",
    "exchange_charge": "Customer asks about fees charged for currency exchange.",
    "exchange_rate": "Customer asks a general question about how exchange rates are determined.",
    "exchange_via_app": "Customer asks how to exchange currency through the mobile app.",
    "extra_charge_on_statement": "Customer sees an unexpected fee or charge on their statement, not tied to a specific card payment or withdrawal.",
    "failed_transfer": "A transfer attempt failed due to an error before completion.",
    "fiat_currency_support": "Asks which traditional (fiat) currencies the service supports.",
    "get_disposable_virtual_card": "Customer asks about, or how to obtain, a single-use disposable virtual card (uses, availability, how it works).",
    "get_physical_card": "Customer asks how to obtain a physical card.",
    "getting_spare_card": "Customer wants an additional or spare physical card on the same account.",
    "getting_virtual_card": "Customer asks how to obtain a virtual card or where their virtual card is.",
    "lost_or_stolen_card": "Customer has physically lost their card or had it stolen.",
    "lost_or_stolen_phone": "Customer has lost their phone or had it stolen and is concerned about app access or linked services.",
    "order_physical_card": "Customer asks about or wants to place an order for a physical card (the order itself, delivery, fees).",
    "passcode_forgotten": "Customer has forgotten their app passcode or password.",
    "pending_card_payment": "A card payment is showing as pending and the customer is asking when it will clear.",
    "pending_cash_withdrawal": "A cash withdrawal is showing as pending and the customer wants to know why or when it will complete.",
    "pending_top_up": "A top-up is showing as pending and not yet credited to the account.",
    "pending_transfer": "A transfer is in pending status and has not yet completed.",
    "pin_blocked": "Customer's PIN is blocked, typically after too many incorrect attempts.",
    "receiving_money": "Customer asks how someone can send them money or what to do once money is received.",
    "Refund_not_showing_up": "Customer is expecting a refund that has not yet appeared on their account.",
    "request_refund": "Customer wants to request a refund on a payment or purchase.",
    "reverted_card_payment?": "A card payment was reverted or reversed and the customer is asking about it.",
    "supported_cards_and_currencies": "Asks which card types or currencies the service supports.",
    "terminate_account": "Customer wants to close or terminate their account.",
    "top_up_by_bank_transfer_charge": "Asks about fees for topping up the account via bank transfer.",
    "top_up_by_card_charge": "Asks about fees for topping up the account via card.",
    "top_up_by_cash_or_cheque": "Asks whether or how the account can be topped up using cash or cheque.",
    "top_up_failed": "A top-up attempt failed and the customer is asking why.",
    "top_up_limits": "Asks about limits (maximum amount, frequency) on top-ups.",
    "top_up_reverted": "A top-up was reversed or cancelled after appearing to succeed.",
    "topping_up_by_card": "Customer asks how to top up using a card, or about top-up by card more generally.",
    "transaction_charged_twice": "Customer was charged twice for the same transaction (duplicate charge).",
    "transfer_fee_charged": "Customer was charged a fee on a transfer and asks why or how much.",
    "transfer_into_account": "Customer asks how to receive a transfer into their account.",
    "transfer_not_received_by_recipient": "Customer sent a transfer but the recipient has not received the funds (transfer is no longer pending on the sender side).",
    "transfer_timing": "Customer asks how long a transfer typically takes.",
    "unable_to_verify_identity": "The identity verification process failed and the customer needs help.",
    "verify_my_identity": "Customer asks about the steps or documents needed to verify their identity.",
    "verify_source_of_funds": "Customer is being asked to, or asks about, verifying the source of their funds.",
    "verify_top_up": "Customer needs to verify a specific top-up that is pending verification.",
    "virtual_card_not_working": "A virtual card is failing to make payments.",
    "visa_or_mastercard": "Customer asks about, or expresses a preference for, Visa vs Mastercard.",
    "why_verify_identity": "Customer asks why identity verification is required.",
    "wrong_amount_of_cash_received": "Customer received less or more cash than expected from an ATM withdrawal.",
    "wrong_exchange_rate_for_cash_withdrawal": "Customer believes an incorrect exchange rate was applied to a cash withdrawal abroad.",
}


def assert_descriptions_complete(
    id_to_label: dict[int, str],
    descriptions: dict[str, str] = BANKING77_DESCRIPTIONS,
) -> None:
    """Raise if any class in `id_to_label` lacks a description."""
    missing = sorted(set(id_to_label.values()) - set(descriptions))
    if missing:
        raise ValueError(f"Missing descriptions for classes: {missing}")


def format_class_list(
    id_to_label: dict[int, str],
    descriptions: dict[str, str] = BANKING77_DESCRIPTIONS,
) -> str:
    """Render the taxonomy as a bullet block for the teacher prompt.

    Output is sorted by class id for determinism. The teacher selects an
    intent by *name*, not id, so the order is purely for readability.
    """
    assert_descriptions_complete(id_to_label, descriptions)
    lines = [
        f"- {id_to_label[label_id]}: {descriptions[id_to_label[label_id]]}"
        for label_id in sorted(id_to_label)
    ]
    return "\n".join(lines)
