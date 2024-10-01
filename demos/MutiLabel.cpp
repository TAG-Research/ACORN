#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <stack>
#include <regex>
#include "utils.cpp"

// Example usage:

int main() {
    std::vector<std::string> query_labels = {
        "CAT=ExteriorAccessories&RATING=4|RATING=5",
        "CAT=ExteriorAccessories&RATING=5",
        "CAT=Automotive&RATING=4|RATING=5&CAT=ReplacementParts|CAT=ExteriorAccessories",
        "CAT=ReplacementParts&RATING=5",
        // Add more query labels as needed
    };

    std::vector<std::string> base_labels = {
        "BRAND=Caltric,CAT=Automotive,CAT=MotorcyclePowersports,CAT=Parts,CAT=Filters,CAT=OilFilters,RATING=5",
        "BRAND=APL,CAT=Automotive,CAT=TiresWheels,CAT=AccessoriesParts,CAT=LugNutsAccessories,CAT=LugNuts,RATING=4",
        "BRAND=Cardone,CAT=Automotive,CAT=ReplacementParts,CAT=BrakeSystem,CAT=CalipersParts,CAT=CaliperBrackets,RATING=5",
        "BRAND=Monroe,CAT=Automotive,CAT=ReplacementParts,CAT=ShocksStrutsSuspension,CAT=Stabilizers,RATING=5",
        "BRAND=SEGADEN,CAT=Automotive,CAT=ExteriorAccessories,RATING=4",
        // Add more base labels as needed
    };

    for ( auto& query_label_str : query_labels) {
        MultiLabel query_label = MultiLabel::fromQuery(query_label_str);
        std::cout << "Query Label: " << query_label_str << std::endl;
        std::cout << "Preprocessed Query: " << std::endl;
        query_label.printQuery();
        for ( auto& base_label_str : base_labels) {
            MultiLabel base_ml = MultiLabel::fromBase(base_label_str);
            bool result = query_label.isSubsetOf(base_ml);
            std::cout << "Base Label: " << base_label_str << std::endl;
            std::cout << "Is Subset: " << (result ? "True" : "False") << "\n" << std::endl;
        }
        std::cout << "-----------------------------------\n" << std::endl;
    }

    return 0;
}
