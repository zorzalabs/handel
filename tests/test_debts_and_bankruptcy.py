import pytest
from helpers import card_id, events, pos, setup

from handel.engine import (
    EndTurn,
    IllegalAction,
    Mortgage,
    Phase,
    SellBuilding,
    apply,
    legal_actions,
)
from handel.engine.board import BOARD
from handel.engine.cards import Deck

EMMEN = pos("Emmen")


def test_raise_funds_when_cash_is_short():
    s = setup(dice=[(2, 3)], owners={"Emmen": 1, "Lelystad": 0}, cash=[5, 1500])
    assert s.phase == Phase.RAISE_FUNDS and s.decision_player == 0
    assert legal_actions(s, 0) == [Mortgage(pos("Lelystad"))]
    with pytest.raises(IllegalAction):
        apply(s, EndTurn())
    apply(s, Mortgage(pos("Lelystad")))
    rent = BOARD[EMMEN].rents[0]
    assert s.players[0].cash == 5 + 30 - rent
    assert s.players[1].cash == 1500 + rent
    assert s.phase == Phase.MANAGE and s.current == 0


def test_raise_funds_must_sell_buildings_before_mortgaging():
    s = setup(dice=[(2, 3)], positions=[0, 0], cash=[0, 1500],
              owners={"Emmen": 1, "Lelystad": 0, "Radom": 0}, houses={"Lelystad": 1, "Radom": 1})
    assert s.phase == Phase.RAISE_FUNDS
    acts = legal_actions(s, 0)
    assert set(acts) == {SellBuilding(1), SellBuilding(2)}
    apply(s, SellBuilding(1))  # +20 covers rent of 7
    assert s.phase == Phase.MANAGE
    assert s.players[0].cash == 20 - 7


def test_bankruptcy_when_assets_cannot_cover_debt_ends_two_player_game():
    s = setup(dice=[(2, 3)], owners={"Emmen": 1}, cash=[5, 1500])
    assert s.phase == Phase.GAME_OVER
    assert s.players[0].bankrupt and s.winner == 1 and s.end_reason == "bankruptcy"
    assert s.players[1].cash == 1505
    assert s.bankrupt_order == [0]


def test_bankruptcy_to_player_transfers_everything():
    lel, rad = pos("Lelystad"), pos("Radom")
    jail_card = card_id(Deck.KANS, "notary")

    def prep(s):
        s.decks[Deck.KANS].remove(jail_card)
        s.players[0].jail_cards.append(jail_card)

    s = setup(num_players=3, dice=[(2, 3), (4, 6)], cash=[0, 1500, 1500], prepare=prep,
              owners={"Emmen": 1, "Kielce": 1, "Apeldoorn": 1, "Lelystad": 0, "Radom": 0,
                      "Toruń": 0},
              houses={"Emmen": 5, "Lelystad": 1, "Radom": 1}, mortgaged=["Toruń"])
    assert s.players[0].bankrupt
    assert s.owner[lel] == 1 and s.owner[rad] == 1 and s.owner[pos("Toruń")] == 1
    assert s.mortgaged[pos("Toruń")]  # mortgages carry over
    assert s.houses[lel] == 0 and s.houses[rad] == 0
    assert s.players[1].cash == 1500 + 2 * 20  # proceeds of selling two houses
    assert s.players[1].jail_cards == [jail_card]
    assert s.current == 1 and s.phase != Phase.GAME_OVER  # game continues with 2 players
    assert events(s, "bankrupt")[0]["creditor"] == 1


def test_bankruptcy_to_bank_returns_properties():
    s = setup(num_players=3, dice=[(4, 5), (4, 6)], cash=[0, 1500, 1500],
              owners={"Lelystad": 0}, mortgaged=["Lelystad"])
    assert s.players[0].bankrupt
    assert s.owner[pos("Lelystad")] == -1 and not s.mortgaged[pos("Lelystad")]
    assert s.current == 1


def test_other_players_may_have_to_raise_funds_on_a_card():
    s = setup(num_players=3, dice=[(2, 4)], cash=[1500, 5, 1500], owners={"Lelystad": 1},
              top_cards={Deck.SKARBIEC: card_id(Deck.SKARBIEC, "name day")})
    assert s.phase == Phase.RAISE_FUNDS and s.decision_player == 1
    apply(s, Mortgage(pos("Lelystad")))
    assert s.players[1].cash == 5 + 30 - 10
    assert s.players[2].cash == 1490
    assert s.players[0].cash == 1520
    assert s.phase == Phase.MANAGE and s.decision_player == 0


def test_bankrupt_player_is_skipped_in_turn_order():
    s = setup(num_players=3, dice=[(4, 5), (4, 6), (4, 6)], cash=[0, 1500, 1500],
              positions=[0, 10, 10])
    assert s.players[0].bankrupt and s.current == 1
    apply(s, EndTurn())
    assert s.current == 2
    apply(s, EndTurn())
    assert s.current == 1 and s.round == 1
