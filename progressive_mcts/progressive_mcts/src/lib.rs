pub mod cost_set;
pub mod klucb;
use serde::Deserialize;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum CostBoundMode {
    Classic,
    Expectimax,
    LowerBound,
    Marginal,
    Same,
}

impl std::fmt::Display for CostBoundMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Classic => write!(f, "classic"),
            Self::Expectimax => write!(f, "expectimax"),
            Self::LowerBound => write!(f, "lower_bound"),
            Self::Marginal => write!(f, "marginal"),
            Self::Same => write!(f, "same"),
        }
    }
}

impl std::str::FromStr for CostBoundMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "classic" => Ok(Self::Classic),
            "expectimax" => Ok(Self::Expectimax),
            "lower_bound" => Ok(Self::LowerBound),
            "marginal" => Ok(Self::Marginal),
            "same" => Ok(Self::Same),
            _ => Err(format!("Invalid CostBoundMode '{}'", s)),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ChildSelectionMode {
    UCB,
    UCBV,
    UCBd,
    KLUCB,
    #[serde(rename = "klucb+")]
    KLUCBP,
    Uniform,
}

impl std::fmt::Display for ChildSelectionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UCB => write!(f, "ucb"),
            Self::UCBV => write!(f, "ucbv"),
            Self::UCBd => write!(f, "ucbd"),
            Self::KLUCB => write!(f, "klucb"),
            Self::KLUCBP => write!(f, "klucb+"),
            Self::Uniform => write!(f, "uniform"),
        }
    }
}

impl std::str::FromStr for ChildSelectionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "ucb" => Ok(Self::UCB),
            "ucbv" => Ok(Self::UCBV),
            "ucbd" => Ok(Self::UCBd),
            "klucb" => Ok(Self::KLUCB),
            "klucb+" => Ok(Self::KLUCBP),
            "uniform" => Ok(Self::Uniform),
            _ => Err(format!("Invalid ChildSelectionMode '{}'", s)),
        }
    }
}
