/**
 * Implements an abstract domain.
 *
 * @file abstract_domain.c
 */
#include "abstract_domain.h"

void abstract_domain_print(
    const AbstractDomain abstract_domain,
    FILE *stream
) {
    switch (abstract_domain.type) {
    case DOMAIN_INTERVAL:
        fprintf(stream, "Interval Abstract Domain");
        break;

    case DOMAIN_HYPERRECTANGLE:
        fprintf(stream, "Hyperrectangle Abstract Domain");
        break;
    }
}
